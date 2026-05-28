"""
sample_chords.py — Sample chord progressions from a trained Chordonomicon GPT.

Adapted from nanoGPT's sample.py with three changes for this model:
  1. WORD-LEVEL tokenization. Our tokens are whitespace-separated strings like
     "<genre:pop_rock>" and "Amin", not characters. encode/decode use split()
     and join-with-spaces accordingly. (Stock sample.py is char-level and would
     KeyError on the first prompt token.)
  2. CONDITIONING PROMPTS. The prompt IS the conditioning. We build it from a
     genre + decade (+ optional seed chords / structure tags) rather than "\n".
  3. <unk> SUPPRESSION. The model can emit <unk> (a real trained token); that's
     meaningless to a musician. We forbid it (and optionally <bos>) at sampling
     by masking those ids out of the logits, without editing model.py.

Three modes, selected with --mode:
  generate : genre+decade only -> a full progression from scratch
  extend   : genre+decade + seed chords -> continues the progression
  section  : genre+decade + a section (e.g. a verse) -> generates what follows
             (e.g. a chorus). Identical mechanism to extend; the seed simply
             ends at a section boundary.

Examples:
  python sample_chords.py --out_dir=out-chords --mode=generate \
      --genre=pop --decade=2010

  python sample_chords.py --out_dir=out-chords --mode=extend \
      --genre=rock --decade=1990 --seed_chords="Emin C G D"

  python sample_chords.py --out_dir=out-chords --mode=section \
      --genre=pop --decade=2010 \
      --seed_chords="<verse> C G Amin F C G Amin F" --next_section=chorus
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# ----------------------------------------------------------------------------- 
# Defaults (override from the command line via configurator.py, e.g. --genre=rock)
# ----------------------------------------------------------------------------- 
init_from = 'resume'        # 'resume' from out_dir; gpt2 variants not used here
out_dir = 'out-chords'      # where the trained checkpoint lives

# --- conditioning / prompt ---
mode = 'generate'           # 'generate' | 'extend' | 'section'
genre = 'pop'               # human-readable genre; spaces are slugified to match vocab
decade = 2010               # integer decade, e.g. 1990, 2000, 2010, 2020
seed_chords = ''            # for extend/section: space-separated chords and/or tags
next_section = ''           # for section mode: append "<{next_section}>" after the seed
                            # (leave empty to let the model decide the next section)

# --- sampling ---
num_samples = 5             # how many progressions to draw
max_new_tokens = 256        # cap on generated tokens (block_size is 256)
temperature = 0.8           # <1 = safer/more repetitive, >1 = more adventurous
top_k = 100                 # keep only the top_k candidates at each step
forbid_unk = True           # mask <unk> so it can never be generated
forbid_bos = True           # mask <bos> so a sample can't start a brand-new song
seed = 1337
device = 'cuda'             # 'cpu' | 'cuda' | 'mps'
dtype = 'float32'           # 'float32' is safest for short CPU/MPS sampling runs
compile = False
exec(open('configurator.py').read())  # let --flags override the above
# ----------------------------------------------------------------------------- 

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
device_type = 'cuda' if 'cuda' in device else ('mps' if device == 'mps' else 'cpu')
ptdtype = {'float32': torch.float32,
           'bfloat16': torch.bfloat16,
           'float16': torch.float16}[dtype]
# autocast only matters on cuda; keep it a nullcontext elsewhere.
ctx = (torch.amp.autocast(device_type='cuda', dtype=ptdtype)
       if device_type == 'cuda' else nullcontext())

# --- load model ----------------------------------------------------------------
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# --- load vocab (meta.pkl) -----------------------------------------------------
# Prefer the dataset the checkpoint was trained on; fall back to data/chords.
meta_path = None
if 'config' in checkpoint and 'dataset' in checkpoint['config']:
    cand = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    if os.path.exists(cand):
        meta_path = cand
if meta_path is None:
    cand = os.path.join('data', 'chords', 'meta.pkl')
    if os.path.exists(cand):
        meta_path = cand
if meta_path is None:
    raise FileNotFoundError(
        "Could not find meta.pkl (looked in the checkpoint's dataset dir and "
        "data/chords). This model needs its chord vocabulary.")
print(f"Loading vocab from {meta_path}")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

# WORD-LEVEL encode/decode (this is the key difference from stock sample.py).
def encode(s: str) -> list[int]:
    ids = []
    for tok in s.split():
        if tok not in stoi:
            raise KeyError(
                f"Prompt token {tok!r} is not in the vocabulary. "
                f"Check spelling / slug (e.g. 'pop rock' -> '<genre:pop_rock>').")
        ids.append(stoi[tok])
    return ids

def decode(ids: list[int]) -> str:
    return ' '.join(itos[i] for i in ids)

# --- build the conditioning prompt ---------------------------------------------
def genre_token(g: str) -> str:
    # Mirror the cleaning script's slugify so 'pop rock' -> '<genre:pop_rock>'.
    return f"<genre:{g.replace(' ', '_')}>"

prompt_parts = ['<bos>', genre_token(genre), f"<decade:{int(decade)}>"]

if mode in ('extend', 'section'):
    if seed_chords.strip():
        prompt_parts.append(seed_chords.strip())
    if mode == 'section' and next_section.strip():
        prompt_parts.append(f"<{next_section.strip()}>")
elif mode != 'generate':
    raise ValueError(f"Unknown mode {mode!r}; use generate|extend|section.")

prompt = ' '.join(prompt_parts)
print(f"\nMode: {mode}")
print(f"Prompt: {prompt}\n")

try:
    start_ids = encode(prompt)
except KeyError as e:
    # Most common cause: a genre/decade not present in this trained vocab.
    avail_genres = sorted(t for t in stoi if t.startswith('<genre:'))
    avail_decades = sorted(t for t in stoi if t.startswith('<decade:'))
    print("ERROR:", e)
    print("Available genres:", avail_genres)
    print("Available decades:", avail_decades)
    raise SystemExit(1)

x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# --- ids we forbid the sampler from emitting -----------------------------------
forbidden_ids = []
if forbid_unk and '<unk>' in stoi:
    forbidden_ids.append(stoi['<unk>'])
if forbid_bos and '<bos>' in stoi:
    forbidden_ids.append(stoi['<bos>'])
eos_id = stoi.get('<eos>', None)

# --- custom generation loop ----------------------------------------------------
# We don't call model.generate() because we need two behaviors it doesn't offer
# out of the box: (a) mask forbidden tokens from the logits, and (b) stop early
# at <eos>. This keeps model.py untouched (minimal-divergence) while giving clean
# output. The sampling math (temperature, top_k, softmax) mirrors nanoGPT's.
@torch.no_grad()
def generate(idx: torch.Tensor) -> torch.Tensor:
    for _ in range(max_new_tokens):
        # crop context to block_size if needed
        idx_cond = idx if idx.size(1) <= model.config.block_size \
            else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        for fid in forbidden_ids:
            logits[:, fid] = float('-inf')
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break  # natural end of progression
    return idx

# --- draw samples --------------------------------------------------------------
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = generate(x)
            text = decode(y[0].tolist())
            # Tidy display: drop <bos>, mark the end clearly.
            text = text.replace('<bos> ', '').replace(' <eos>', '  [end]')
            print(f"=== sample {k + 1} ===")
            print(text)
            print()
