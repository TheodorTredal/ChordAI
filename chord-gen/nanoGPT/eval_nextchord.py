"""
eval_nextchord.py — Next-chord prediction accuracy for trained chord models.

Measures how well a trained model predicts the NEXT CHORD given the true
preceding context (teacher-forced), on the validation set. This is the
Chordonomicon paper's metric and the apples-to-apples way to compare models
(v1 vs v2 vs the future RNN baseline) — unlike raw val loss, which is not
comparable across v1/v2 because v2 masks conditioning tokens out of its loss.

What it reports:
  * top-1 and top-3 accuracy
  * aggregate AND per-genre (per-genre is where v2's stronger conditioning,
    if real, should show up — especially for rarer genres)
  * CHORD tokens only: predictions of structure tags (<verse> etc.),
    conditioning tokens, <bos>/<eos>/<unk> are excluded from the denominator,
    because "can it predict the next chord" is the musical question. Including
    trivially-predictable structure tokens would inflate every model equally
    and muddy the comparison.

Methodology (identical for every model, so only the weights differ):
  For each validation song we build the true token sequence
      <bos> <genre> <decade> <... chords/sections ...> <eos>
  feed it teacher-forced, and at each position t whose TARGET (t+1) is a chord
  token, check whether the true chord is the model's top-1 / within top-3.
  The model always sees the true, full prefix (incl. genre+decade) — which is
  also the real inference condition.

Usage:
  python eval_nextchord.py --out_dir=out-chords-v1 --data-dir=data/chords
  python eval_nextchord.py --out_dir=out-chords-v2 --data-dir=data/chords

  # convenience: evaluate two checkpoints and print a side-by-side table
  python eval_nextchord.py --compare out-chords-v1 out-chords-v2 --data-dir=data/chords

Notes:
  * Reads the cleaned val split from the parquet (so we have per-song genre
    labels) + vocab from meta.pkl. Point --clean-dir at the clean_chordonomicon
    output if it differs from the nanoGPT data dir.
  * Uses the SAME stratified split + seed as prepare.py so "val" here matches
    the songs the model never trained on. Keep --val-frac/--seed in sync with
    the prepare.py run that built the model's data.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from model_loader import load_any


def stratified_val_df(df, val_frac, seed):
    """Reproduce prepare.py's stratified-by-genre split and return ONLY val.

    Must match prepare.py's logic exactly so the val songs here are the same
    ones held out from training.
    """
    rng = np.random.default_rng(seed)
    val_parts = []
    for genre, group in df.groupby("main_genre", sort=True):
        idx = rng.permutation(len(group))
        n_val = int(round(len(group) * val_frac))
        n_val = min(max(n_val, 1), len(group) - 1) if len(group) > 1 else 0
        val_parts.append(group.iloc[idx].iloc[:n_val])
    return pd.concat(val_parts)


@torch.no_grad()
def evaluate(model, val_df, stoi, device, block_size, chord_ids,
             max_songs=None, batch_report_every=2000):
    """Teacher-forced next-chord top-1/top-3 accuracy, aggregate + per-genre."""
    agg = {'top1': 0, 'top3': 0, 'n': 0}
    per_genre = defaultdict(lambda: {'top1': 0, 'top3': 0, 'n': 0})
    unk_id = stoi.get('<unk>')

    songs = val_df.itertuples(index=False)
    total = len(val_df) if max_songs is None else min(max_songs, len(val_df))

    for si, row in enumerate(songs):
        if max_songs is not None and si >= max_songs:
            break
        genre = row.main_genre
        # Encode the full true sequence: <bos> + sequence tokens + <eos>.
        toks = ['<bos>'] + row.sequence.split() + ['<eos>']
        ids = [stoi.get(t, unk_id) for t in toks]
        # Truncate to block_size (model can't see further back than its context).
        # We evaluate predictions within the first block_size positions.
        ids = ids[:block_size]
        if len(ids) < 2:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None, :]
        targets = ids[1:]  # target[t] is the true token after position t

        # IMPORTANT: nanoGPT's GPT.forward only computes logits for the LAST
        # position when called without targets (an inference speed optimization),
        # returning shape (1, 1, vocab). We need logits at EVERY position, so we
        # pass a targets tensor — that puts forward on the full-logits branch
        # returning (1, T, vocab). (The returned loss is ignored here.) The
        # target values themselves don't affect the logits; we score manually
        # below. ignore_index=-1 isn't needed since we don't use the loss.
        y = torch.tensor(targets, dtype=torch.long, device=device)[None, :]
        logits, _ = model(x, y)          # (1, T, vocab)
        logits = logits[0]               # (T, vocab)
        # top-3 indices per position
        top3 = torch.topk(logits, k=3, dim=-1).indices  # (T, 3)
        top1 = top3[:, 0]

        for t, true_id in enumerate(targets):
            # Only score positions whose TRUE NEXT token is a chord.
            if true_id not in chord_ids:
                continue
            is1 = int(top1[t].item() == true_id)
            is3 = int(true_id in top3[t].tolist())
            agg['top1'] += is1; agg['top3'] += is3; agg['n'] += 1
            g = per_genre[genre]
            g['top1'] += is1; g['top3'] += is3; g['n'] += 1

        if batch_report_every and (si + 1) % batch_report_every == 0:
            print(f"  ... {si + 1}/{total} songs", flush=True)

    return agg, per_genre


def pct(a, b):
    return 100.0 * a / b if b else float('nan')


def run_one(out_dir, val_df, stoi, device, chord_ids, max_songs):
    model, ckpt, model_type = load_any(out_dir, device)
    block_size = ckpt['model_args']['block_size']
    print(f"Evaluating {out_dir} [{model_type}] (block_size={block_size}, "
          f"iter={ckpt.get('iter_num','?')}) on {len(val_df):,} val songs ...")
    agg, per_genre = evaluate(model, val_df, stoi, device, block_size,
                              chord_ids, max_songs=max_songs)
    return agg, per_genre


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=None,
                   help="Checkpoint dir to evaluate (single-model mode).")
    p.add_argument('--compare', nargs=2, default=None, metavar=('DIR_A', 'DIR_B'),
                   help="Two checkpoint dirs for a side-by-side table.")
    p.add_argument('--data-dir', default='data/chords',
                   help="nanoGPT data dir holding meta.pkl.")
    p.add_argument('--clean-dir', default='cleaned',
                   help="Dir holding chordonomicon_clean.parquet (+ vocab.json).")
    p.add_argument('--val-frac', type=float, default=0.1,
                   help="MUST match the prepare.py run that built the model.")
    p.add_argument('--seed', type=int, default=42,
                   help="MUST match the prepare.py run that built the model.")
    p.add_argument('--max-songs', type=int, default=None,
                   help="Cap val songs for a quick run (default: all).")
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    # --- vocab + chord-id set ------------------------------------------------
    with open(os.path.join(args.data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    # Chord tokens = everything that is NOT a <...> tag and not a special token.
    specials = {'<bos>', '<eos>', '<unk>', '<pad>'}
    chord_ids = set()
    for tok, tid in stoi.items():
        if tok in specials:
            continue
        if tok.startswith('<') and tok.endswith('>'):
            continue  # genre/decade/section tags
        chord_ids.add(tid)
    print(f"Chord vocabulary scored: {len(chord_ids):,} chord tokens")

    # --- rebuild the val split ----------------------------------------------
    parquet = os.path.join(args.clean_dir, 'chordonomicon_clean.parquet')
    df = pd.read_parquet(parquet)
    val_df = stratified_val_df(df, args.val_frac, args.seed)
    print(f"Reconstructed val split: {len(val_df):,} songs "
          f"(val_frac={args.val_frac}, seed={args.seed})\n")

    dirs = args.compare if args.compare else [args.out_dir]
    if dirs == [None]:
        p.error("Provide --out_dir DIR or --compare DIR_A DIR_B")

    results = {}
    for d in dirs:
        agg, per_genre = run_one(d, val_df, stoi, args.device, chord_ids,
                                 args.max_songs)
        results[d] = (agg, per_genre)
        print(f"\n=== {d} ===")
        print(f"  AGGREGATE  top1 {pct(agg['top1'],agg['n']):5.2f}%   "
              f"top3 {pct(agg['top3'],agg['n']):5.2f}%   "
              f"(n={agg['n']:,} chord predictions)")
        print("  per genre:")
        for g in sorted(per_genre, key=lambda k: -per_genre[k]['n']):
            s = per_genre[g]
            print(f"    {g:<14} top1 {pct(s['top1'],s['n']):5.2f}%   "
                  f"top3 {pct(s['top3'],s['n']):5.2f}%   (n={s['n']:,})")
        print()

    # --- side-by-side comparison table --------------------------------------
    if args.compare:
        a, b = args.compare
        (agg_a, pg_a), (agg_b, pg_b) = results[a], results[b]
        print("=" * 72)
        print(f"COMPARISON  (A = {a}   B = {b})")
        print("=" * 72)
        print(f"{'genre':<14}{'A top1':>9}{'B top1':>9}{'Δtop1':>8}"
              f"{'A top3':>9}{'B top3':>9}{'Δtop3':>8}")
        rows = ['__AGG__'] + sorted(pg_a, key=lambda k: -pg_a[k]['n'])
        for g in rows:
            if g == '__AGG__':
                sa, sb, label = agg_a, agg_b, 'AGGREGATE'
            else:
                sa, sb, label = pg_a[g], pg_b.get(g, {'top1':0,'top3':0,'n':0}), g
            a1, b1 = pct(sa['top1'],sa['n']), pct(sb['top1'],sb['n'])
            a3, b3 = pct(sa['top3'],sa['n']), pct(sb['top3'],sb['n'])
            print(f"{label:<14}{a1:>8.2f}{b1:>9.2f}{b1-a1:>+8.2f}"
                  f"{a3:>9.2f}{b3:>9.2f}{b3-a3:>+8.2f}")
        print("\nΔ > 0 means B (the second dir) is better than A.")


if __name__ == '__main__':
    main()
