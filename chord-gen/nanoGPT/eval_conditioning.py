"""
eval_conditioning.py — Does the model's output actually depend on the genre?

Next-chord accuracy (eval_nextchord.py) is measured teacher-forced, with the
true local chord context always present, so it barely exercises the genre/decade
conditioning. THIS script tests the opposite regime: FREE GENERATION from just
<bos> <genre> <decade>, where the conditioning is the only signal. It asks: do
different genres produce DISTINGUISHABLE chord distributions, or does the model
generate the same chord-soup regardless of the genre token?

Metric: between-genre separation of generated chord distributions.
  * For each genre, generate N progressions (decade held FIXED to isolate genre)
    and build a normalized histogram over chord tokens.
  * Pairwise Jensen-Shannon divergence between genres (0 = identical dists,
    1 = disjoint). Mean pairwise JSD = one "genre separation" score per model.
  * A model whose conditioning truly drives output -> higher separation.
    A model that ignores the genre token -> genres bunch together -> low score.

Also reports interpretable musical features per genre (share of seventh /
diminished / extended / slash chords, mean progression length, <eos> rate), so
"jazz has more 7ths than punk" is legible without reading divergence numbers.

Runs against any checkpoint; --compare does two side by side. Decade and
temperature are held fixed and identical across models so only weights differ.

Usage:
  python eval_conditioning.py --out_dir=out-CHORDv2 --data-dir=data/chords
  python eval_conditioning.py --compare out-CHORDv1 out-CHORDv2 --data-dir=data/chords \
      --n-per-genre=300 --decade=2010 --temperature=1.0
"""

import argparse
import math
import os
import pickle
from collections import Counter, defaultdict

import torch

from model_loader import load_any


# ----------------------------------------------------------------------------- 
# Model loading / generation
# ----------------------------------------------------------------------------- 

def load_model(out_dir, device):
    model, ckpt, _ = load_any(out_dir, device)
    return model, ckpt


@torch.no_grad()
def generate_batch(model, prompt_ids, n, max_new_tokens, temperature, top_k,
                   eos_id, forbid_ids, device, block_size):
    """Generate n sequences from the same prompt. Returns list of id-lists
    (generated continuation only, excluding the prompt). Batched for speed."""
    out = []
    B = 64  # sub-batch size
    for start in range(0, n, B):
        bs = min(B, n - start)
        idx = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
        idx = idx.repeat(bs, 1)
        finished = torch.zeros(bs, dtype=torch.bool, device=device)
        gen = [[] for _ in range(bs)]
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            logits, _ = model(idx_cond, None)   # last-position logits (1-step)
            logits = logits[:, -1, :] / temperature
            for fid in forbid_ids:
                logits[:, fid] = float('-inf')
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)         # (bs,1)
            idx = torch.cat([idx, nxt], dim=1)
            for b in range(bs):
                if finished[b]:
                    continue
                tok = nxt[b, 0].item()
                if tok == eos_id:
                    finished[b] = True
                else:
                    gen[b].append(tok)
            if finished.all():
                break
        out.extend(gen)
    return out


# ----------------------------------------------------------------------------- 
# Chord-feature helpers (work on token STRINGS)
# ----------------------------------------------------------------------------- 

def is_chord(tok):
    return not (tok.startswith('<') and tok.endswith('>'))

def chord_features(tok):
    """Return a set of feature flags for a chord token string."""
    f = set()
    # seventh chords: '7' appears (maj7, min7, dom7, etc.)
    if '7' in tok:
        f.add('seventh')
    # diminished / augmented
    if 'dim' in tok:
        f.add('dim')
    if 'aug' in tok:
        f.add('aug')
    # extended/added tensions
    if any(s in tok for s in ('9', '11', '13', 'add', 'sus')):
        f.add('extended')
    # slash / inversion
    if '/' in tok:
        f.add('slash')
    return f


# ----------------------------------------------------------------------------- 
# Distribution distance
# ----------------------------------------------------------------------------- 

def js_divergence(p, q):
    """Jensen-Shannon divergence between two dicts of {token: prob}. Base-2,
    so result is in [0, 1]."""
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    def kl(a, b):
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            if ak > 0 and b.get(k, 0.0) > 0:
                s += ak * math.log2(ak / b[k])
        return s
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def normalize(counter):
    tot = sum(counter.values())
    return {k: v / tot for k, v in counter.items()} if tot else {}


# ----------------------------------------------------------------------------- 
# Per-model evaluation
# ----------------------------------------------------------------------------- 

def evaluate_model(out_dir, genres, stoi, itos, device, args):
    model, ckpt = load_model(out_dir, device)
    block_size = ckpt['model_args']['block_size']
    bos, eos = stoi['<bos>'], stoi['<eos>']
    forbid = [stoi[t] for t in ('<unk>', '<bos>') if t in stoi]
    decade_tok = f'<decade:{args.decade}>'
    if decade_tok not in stoi:
        raise SystemExit(f"{decade_tok} not in vocab; pick a present decade.")

    genre_hist = {}        # genre -> normalized chord histogram
    genre_feats = {}       # genre -> feature-rate dict
    genre_meta = {}        # genre -> (eos_rate, mean_len)

    print(f"\nGenerating for {out_dir} "
          f"(decade={args.decade}, T={args.temperature}, "
          f"{args.n_per_genre}/genre) ...")
    for g in genres:
        g_slug = g.replace(' ', '_')
        gtok = f'<genre:{g_slug}>'
        if gtok not in stoi:
            print(f"  skip {g}: {gtok} not in vocab"); continue
        prompt = [bos, stoi[gtok], stoi[decade_tok]]
        seqs = generate_batch(model, prompt, args.n_per_genre,
                              args.max_new_tokens, args.temperature,
                              args.top_k, eos, forbid, device, block_size)
        chords = Counter()
        feat_counts = Counter()
        n_chords_total = 0
        eos_hits = 0
        lengths = []
        for s in seqs:
            toks = [itos[i] for i in s]
            cs = [t for t in toks if is_chord(t)]
            lengths.append(len(cs))
            # eos_hit: generation stopped before max_new_tokens -> reached eos
            if len(s) < args.max_new_tokens:
                eos_hits += 1
            for c in cs:
                chords[c] += 1
                n_chords_total += 1
                for ff in chord_features(c):
                    feat_counts[ff] += 1
        genre_hist[g] = normalize(chords)
        genre_feats[g] = {k: (feat_counts[k] / n_chords_total if n_chords_total else 0.0)
                          for k in ('seventh', 'dim', 'aug', 'extended', 'slash')}
        genre_meta[g] = (eos_hits / len(seqs) if seqs else 0.0,
                         sum(lengths) / len(lengths) if lengths else 0.0)
        print(f"  {g:<14} {n_chords_total:>7,} chords  "
              f"eos {genre_meta[g][0]*100:4.0f}%  meanlen {genre_meta[g][1]:5.1f}")

    # mean pairwise JSD across genres = the genre-separation score
    gs = [g for g in genres if g in genre_hist]
    pair_js = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            pair_js.append(js_divergence(genre_hist[gs[i]], genre_hist[gs[j]]))
    mean_js = sum(pair_js) / len(pair_js) if pair_js else 0.0

    return {
        'mean_js': mean_js,
        'genre_hist': genre_hist,
        'genre_feats': genre_feats,
        'genre_meta': genre_meta,
        'genres': gs,
    }


def print_report(out_dir, res):
    print(f"\n=== {out_dir} ===")
    print(f"  GENRE SEPARATION (mean pairwise JSD): {res['mean_js']:.4f}   "
          f"(higher = genres more distinct)")
    print("  per-genre chord-feature rates (share of generated chords):")
    print(f"    {'genre':<14}{'7th':>7}{'dim':>7}{'aug':>7}{'ext':>7}"
          f"{'slash':>7}{'eos%':>7}{'len':>7}")
    for g in res['genres']:
        f = res['genre_feats'][g]
        eos_r, mlen = res['genre_meta'][g]
        print(f"    {g:<14}{f['seventh']*100:>6.1f}{f['dim']*100:>7.1f}"
              f"{f['aug']*100:>7.1f}{f['extended']*100:>7.1f}{f['slash']*100:>7.1f}"
              f"{eos_r*100:>6.0f}{mlen:>7.1f}")


def main():
    p = argparse.ArgumentParser()
    # Positional: 1+ checkpoint directories. With 1 -> single-model report.
    # With 2+ -> N-way side-by-side, first listed is the reference for Δ.
    p.add_argument('out_dirs', nargs='+',
                   help="One or more checkpoint dirs. First listed = reference "
                        "for Δ when multiple are given.")
    p.add_argument('--data-dir', default='data/chords')
    p.add_argument('--n-per-genre', type=int, default=300)
    p.add_argument('--decade', type=int, default=2010)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=0, help="0 disables top-k")
    p.add_argument('--max-new-tokens', type=int, default=200)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1337)
    args = p.parse_args()
    if args.top_k == 0:
        args.top_k = None

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    with open(os.path.join(args.data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    genres = sorted(t[len('<genre:'):-1].replace('_', ' ')
                    for t in stoi if t.startswith('<genre:'))

    dirs = args.out_dirs

    results = {}
    for d in dirs:
        results[d] = evaluate_model(d, genres, stoi, itos, args.device, args)
        # Always print the single-model report (helpful even in N-way mode —
        # gives the full per-genre feature table for each model before the
        # cross-model summary tables below).
        print_report(d, results[d])

    # --- N-way summary tables (only when 2+ models) --------------------------
    if len(dirs) < 2:
        return

    # Short, unique display labels for the table headers.
    labels = [os.path.basename(os.path.normpath(d)) or d for d in dirs]
    seen = {}
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
    if any(v > 1 for v in seen.values()):
        labels = [f"{lab}#{i}" for i, lab in enumerate(labels)]
    ref_label = labels[0]

    print("\n" + "=" * 72)
    print(f"N-WAY COMPARISON  (reference for Δ = {ref_label})")
    print("=" * 72)

    # 1) Genre-separation score (one number per model).
    print("\nGENRE SEPARATION SCORE (mean pairwise JSD; higher = more distinct)")
    ref_js = results[dirs[0]]['mean_js']
    for d, lab in zip(dirs, labels):
        js = results[d]['mean_js']
        marker = "  (ref)" if d == dirs[0] else f"   Δ {js - ref_js:+.4f}"
        print(f"  {lab:<28} {js:.4f}{marker}")

    col_w = max(10, max(len(lab) for lab in labels) + 1)

    # 2) Per-feature side-by-side: rows = genres, cols = models. We print one
    # mini-table per musically-interpretable feature, so a reader can scan e.g.
    # "jazz 7th rate across models" in a single row. This is the layout that
    # makes "feature climbs RNN < v1 < v2" legible — the writeup-ready view.
    feature_labels = [('seventh', '7th'), ('dim', 'dim'),
                      ('extended', 'ext'), ('slash', 'slash')]
    # Use the reference's genre order (sorted alpha for stability across runs).
    ref_genres = sorted(results[dirs[0]]['genres'])
    for fkey, fname in feature_labels:
        print(f"\nPer-genre {fname} chord rate (% of generated chords)")
        head = f"{'genre':<14}" + ''.join(f"{lab:>{col_w}}" for lab in labels)
        print(head)
        for g in ref_genres:
            line = f"{g:<14}"
            for d in dirs:
                v = results[d]['genre_feats'].get(g, {}).get(fkey, 0.0) * 100
                line += f"{v:>{col_w}.1f}"
            print(line)

    # 3) Per-genre meta (<eos> rate + mean length) compactly.
    print(f"\nGeneration completeness: <eos> rate (%) per genre")
    head = f"{'genre':<14}" + ''.join(f"{lab:>{col_w}}" for lab in labels)
    print(head)
    for g in ref_genres:
        line = f"{g:<14}"
        for d in dirs:
            v = results[d]['genre_meta'].get(g, (0.0, 0.0))[0] * 100
            line += f"{v:>{col_w}.0f}"
        print(line)


if __name__ == '__main__':
    main()
