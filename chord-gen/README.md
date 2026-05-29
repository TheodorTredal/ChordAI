# chord-gen

Custom GPT and LSTM models trained on the [Chordonomicon](https://huggingface.co/datasets/ailsntua/Chordonomicon) dataset to generate genre- and era-conditioned chord progressions.

## Quick start

```bash
cd chord-gen
make setup    # create venv + install dependencies
make data     # download dataset and prepare binary files  (~30 min first run)
make train    # train the GPT model                        (~2–4 h on a single GPU)
make sample   # verify the model generates chord progressions
```

Run `make` with no arguments for a summary of all targets.

## Prerequisites

- Python 3.10+
- GNU Make
- A CUDA GPU is strongly recommended. CPU training works but is very slow (see [CPU / MPS](#cpu--mps)).
- ~2 GB free disk space for the dataset files.

---

## Pipeline

### Step 1 — Environment setup

```bash
make setup
```

Creates `venv/` and installs dependencies from `requirements.txt`.

**GPU training:** before running `make setup`, replace the `torch` line in `requirements.txt` with the CUDA wheel for your driver version. Find the right command at <https://pytorch.org/get-started/locally/>.

Example for CUDA 12.1:
```bash
# in requirements.txt, change:
torch>=2.1
# to nothing, then install separately:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 2 — Prepare the dataset

```bash
make data
```

Runs two scripts in sequence:

1. **`dataset/clean_chordonomicon.py`** — streams ~680 k songs directly from the Hugging Face Hub (`ailsntua/Chordonomicon`), cleans them (drops rows missing genre or decade, collapses variant section tags like `<verse_2>` → `<verse>`, builds a vocabulary of the 819 most-frequent chords), and writes `dataset/cleaned/`.

2. **`dataset/prepare.py`** — encodes the cleaned data into the binary token-ID files that `train.py` reads, with a genre-stratified 90/10 train/val split. Writes `nanoGPT/data/chords/`.

Expected outputs (first run takes ~30 min, most of which is the download):

```
dataset/cleaned/
  chordonomicon_clean.parquet   # ~300 k songs after filtering
  vocab.json                    # 851-token vocabulary
  report.txt                    # cleaning statistics

nanoGPT/data/chords/
  train.bin                     # ~23 M training tokens
  val.bin                       # ~2.6 M validation tokens
  meta.pkl                      # vocab mappings read by nanoGPT
  train.offsets.npy             # song-boundary offsets (v2 doc-aware batching)
  val.offsets.npy
```

Subsequent runs skip any step whose output file is already up to date (standard Make behaviour).

---

### Step 3 — Train

**GPT (the model used by the server):**

```bash
make train
```

Trains for up to 20 000 iterations (~14 epochs). The checkpoint with the lowest validation loss is saved to `nanoGPT/out-CHORDv2/ckpt.pt`.

**RNN / LSTM baseline** (for architecture comparison):

```bash
make train-rnn
```

Saves to `nanoGPT/out-CHORDrnn/ckpt.pt`.

#### CPU / MPS

```bash
# CPU
make train TRAIN_ARGS="--device=cpu --compile=False"

# Apple Silicon (MPS)
make train TRAIN_ARGS="--device=mps --compile=False"
```

---

### Step 4 — Sample

```bash
make sample          # quick sanity check: 3 pop/2010 progressions
```

Custom parameters:

```bash
cd nanoGPT
../venv/bin/python sample.py \
    --out_dir=out-CHORDv2 \
    --mode=generate \
    --genre=rock \
    --decade=1990 \
    --num_samples=5 \
    --temperature=0.9
```

**Sampling modes:**

| Mode | Description | Extra flags |
|------|-------------|-------------|
| `generate` | Full progression from genre + decade | — |
| `extend` | Continue a seed progression | `--seed_chords="Am F C G"` |
| `section` | Generate the next section | `--seed_chords="<verse> C G Am F"` + `--next_section=chorus` |

**Available genres:**
`alternative` · `country` · `electronic` · `jazz` · `metal` · `pop` · `pop_rock` · `punk` · `rap` · `reggae` · `rock` · `soul`

**Available decades:** `1950` · `1960` · `1970` · `1980` · `1990` · `2000` · `2010` · `2020`

The server maps common planner genre names (e.g. `rnb` → `soul`, `hiphop` → `rap`, `edm` → `electronic`) automatically — see `server/models/chord_runner.go`.

---

## Model details

| | GPT v1 | GPT v2 | RNN |
|---|---|---|---|
| Architecture | Transformer | Transformer | LSTM |
| Parameters | ~10.7 M | ~10.7 M | ~10.9 M |
| Batching | Random window | Doc-aware (song boundaries) | Random window |
| Prefix masking | No | Yes | No |
| Checkpoint dir | `out-CHORDv1` | `out-CHORDv2` | `out-CHORDrnn` |
| Used by server | No | **Yes** | No |

**v2 improvements over v1:** training windows always start at a song boundary so the conditioning prefix (`<genre:..> <decade:..>`) is visible at position 0. The loss is masked over those conditioning tokens so the model is only penalised on chord prediction, not on reproducing the prompt.

The model architecture itself (`model.py`) is unchanged between v1 and v2 — the difference is entirely in the data loader in `train.py`.

---

## Project structure

```
chord-gen/
├── dataset/
│   ├── clean_chordonomicon.py   # Downloads and cleans raw data from HuggingFace
│   ├── prepare.py               # Encodes cleaned data into nanoGPT binary format
│   └── cleaned/                 # Generated by make data — not committed to git
├── nanoGPT/
│   ├── config/
│   │   └── train_chords.py      # GPT hyperparameters (6-layer, 384-dim, ~10.7 M params)
│   ├── data/chords/             # Generated by make data — not committed to git
│   ├── out-CHORDv1/             # Trained checkpoint — GPT v1
│   ├── out-CHORDv2/             # Trained checkpoint — GPT v2  ← used by the server
│   ├── out-CHORDrnn/            # Trained checkpoint — RNN/LSTM baseline
│   ├── model.py                 # GPT architecture (nanoGPT, minimal changes)
│   ├── model_loader.py          # Unified loader for GPT and RNN checkpoints
│   ├── train.py                 # GPT training loop (v1 + v2 batching modes)
│   ├── train_rnn.py             # RNN/LSTM training
│   └── sample.py                # Chord generation (generate / extend / section modes)
├── Makefile                     # Pipeline entry point
└── requirements.txt             # Python dependencies
```

---

## Integration with the Go server

The Go server calls `chord-gen/nanoGPT/sample.py` as a subprocess and looks for the trained checkpoint at `chord-gen/nanoGPT/out-CHORDv2/ckpt.pt`. If the checkpoint is absent the server falls back to a hardcoded stub that returns genre-appropriate progressions without neural generation.

See [`server/models/chord_runner.go`](../server/models/chord_runner.go) for the integration code.
