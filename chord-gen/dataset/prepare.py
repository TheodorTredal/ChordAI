#!/usr/bin/env python3
"""
prepare.py
==========

Bridge the cleaned Chordonomicon dataset into the binary format nanoGPT's
train.py expects, with zero changes required to train.py or model.py.

Reads:
  - chordonomicon_clean.parquet  (columns: main_genre, decade, sequence, seq_len)
  - vocab.json                   (token_to_id, id_to_token, special-token groups)
both produced by clean_chordonomicon.py.

Writes (into --outdir, default ./data/chords):
  - train.bin   concatenated uint16 token IDs for the training split
  - val.bin     concatenated uint16 token IDs for the validation split
  - meta.pkl    {vocab_size, stoi, itos}  -- read automatically by nanoGPT

Design decisions (minimal-divergence version):
  * Each song is encoded as:  <bos> <genre:..> <decade:..> <...sequence...> <eos>
    The conditioning prefix (genre, decade) is already the first two tokens of
    the 'sequence' field from the cleaning script, so we prepend <bos> and
    append <eos> around the whole thing.
  * Songs are concatenated into one long stream per split (nanoGPT samples
    random windows from it). The <eos>/<bos> boundary tokens are what let the
    model learn where songs end and that a fresh song starts with conditioning.
  * The train/val split is STRATIFIED BY GENRE so rare genres (electronic, etc.)
    appear proportionally in both splits despite heavy class imbalance.
  * No prefix loss-masking and no song-boundary-aware batching -- those are the
    "more-correct" extensions we deliberately deferred. train.py is untouched.

Usage:
  python prepare.py --data-dir ./cleaned --outdir ./data/chords
  python prepare.py --data-dir ./cleaned --val-frac 0.1 --seed 42
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# nanoGPT stores token ids as uint16; assert the vocab fits to fail loudly
# rather than silently overflowing if the vocab ever grows past 65535.
ID_DTYPE = np.uint16
ID_DTYPE_MAX = np.iinfo(ID_DTYPE).max


def stratified_split(df: pd.DataFrame, val_frac: float, seed: int):
    """Split rows into train/val, stratified by main_genre.

    Within each genre we shuffle deterministically and peel off val_frac for
    validation, so every genre is represented in both splits in proportion to
    its size. Returns (train_df, val_df).
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for genre, group in df.groupby("main_genre", sort=True):
        idx = rng.permutation(len(group))
        n_val = int(round(len(group) * val_frac))
        # Guarantee at least 1 val example for any non-empty genre, and never
        # take the entire genre into val.
        n_val = min(max(n_val, 1), len(group) - 1) if len(group) > 1 else 0
        shuffled = group.iloc[idx]
        val_parts.append(shuffled.iloc[:n_val])
        train_parts.append(shuffled.iloc[n_val:])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed)
    return train_df, val_df
#!/usr/bin/env python3
"""
prepare.py
==========

Bridge the cleaned Chordonomicon dataset into the binary format nanoGPT's
train.py expects, with zero changes required to train.py or model.py.

Reads:
  - chordonomicon_clean.parquet  (columns: main_genre, decade, sequence, seq_len)
  - vocab.json                   (token_to_id, id_to_token, special-token groups)
both produced by clean_chordonomicon.py.

Writes (into --outdir, default ./data/chords):
  - train.bin   concatenated uint16 token IDs for the training split
  - val.bin     concatenated uint16 token IDs for the validation split
  - meta.pkl    {vocab_size, stoi, itos}  -- read automatically by nanoGPT

Design decisions (minimal-divergence version):
  * Each song is encoded as:  <bos> <genre:..> <decade:..> <...sequence...> <eos>
    The conditioning prefix (genre, decade) is already the first two tokens of
    the 'sequence' field from the cleaning script, so we prepend <bos> and
    append <eos> around the whole thing.
  * Songs are concatenated into one long stream per split (nanoGPT samples
    random windows from it). The <eos>/<bos> boundary tokens are what let the
    model learn where songs end and that a fresh song starts with conditioning.
  * The train/val split is STRATIFIED BY GENRE so rare genres (electronic, etc.)
    appear proportionally in both splits despite heavy class imbalance.
  * No prefix loss-masking and no song-boundary-aware batching -- those are the
    "more-correct" extensions we deliberately deferred. train.py is untouched.

Usage:
  python prepare.py --data-dir ./cleaned --outdir ./data/chords
  python prepare.py --data-dir ./cleaned --val-frac 0.1 --seed 42
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# nanoGPT stores token ids as uint16; assert the vocab fits to fail loudly
# rather than silently overflowing if the vocab ever grows past 65535.
ID_DTYPE = np.uint16
ID_DTYPE_MAX = np.iinfo(ID_DTYPE).max


def stratified_split(df: pd.DataFrame, val_frac: float, seed: int):
    """Split rows into train/val, stratified by main_genre.

    Within each genre we shuffle deterministically and peel off val_frac for
    validation, so every genre is represented in both splits in proportion to
    its size. Returns (train_df, val_df).
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for genre, group in df.groupby("main_genre", sort=True):
        idx = rng.permutation(len(group))
        n_val = int(round(len(group) * val_frac))
        # Guarantee at least 1 val example for any non-empty genre, and never
        # take the entire genre into val.
        n_val = min(max(n_val, 1), len(group) - 1) if len(group) > 1 else 0
        shuffled = group.iloc[idx]
        val_parts.append(shuffled.iloc[:n_val])
        train_parts.append(shuffled.iloc[n_val:])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed)
    return train_df, val_df


def encode_split(df: pd.DataFrame, stoi: dict, bos_id: int, eos_id: int,
                 unk_id: int) -> tuple[np.ndarray, int, np.ndarray, int]:
    """Encode all songs in a split into one concatenated uint16 array.

    Each song becomes [<bos>] + token_ids + [<eos>]. Returns:
      arr        : the concatenated uint16 token-id stream
      unk_hits   : count of tokens that fell back to <unk> (should be ~0)
      offsets    : start index (into arr) of each song's <bos>. The v2
                   document-aware batcher samples window starts from these so
                   every training window begins at a song (conditioning visible
                   at position 0). Saved as <split>.offsets.npy.
      prefix_len : number of leading tokens forming the conditioning prefix:
                   <bos> <genre:..> <decade:..> = 3. The v2 prefix-masking
                   batcher sets these target positions to -1 so they are
                   excluded from the loss (nanoGPT's cross_entropy already uses
                   ignore_index=-1, so model.py needs no change).
    """
    ids: list[int] = []
    offsets: list[int] = []
    unk_hits = 0
    for seq in df["sequence"]:
        offsets.append(len(ids))       # this song's <bos> position in the stream
        ids.append(bos_id)
        for tok in seq.split():
            tid = stoi.get(tok)
            if tid is None:
                tid = unk_id
                unk_hits += 1
            ids.append(tid)
        ids.append(eos_id)
    arr = np.array(ids, dtype=ID_DTYPE)
    offsets_arr = np.array(offsets, dtype=np.int64)
    # Prefix = <bos> + <genre> + <decade>. seq always starts with the two
    # conditioning tokens (per the cleaning script), so prefix length is 3.
    prefix_len = 3
    return arr, unk_hits, offsets_arr, prefix_len


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert cleaned Chordonomicon data to nanoGPT binaries.")
    p.add_argument("--data-dir", default=Path("./cleaned"), type=Path,
                   help="Dir containing chordonomicon_clean.parquet + vocab.json")
    p.add_argument("--outdir", default=Path("./data/chords"), type=Path,
                   help="Output dir for train.bin/val.bin/meta.pkl "
                        "(default ./data/chords).")
    p.add_argument("--val-frac", default=0.1, type=float,
                   help="Fraction of each genre held out for validation "
                        "(default 0.1).")
    p.add_argument("--seed", default=42, type=int,
                   help="Random seed for the split (default 42).")
    args = p.parse_args(argv)

    parquet_path = args.data_dir / "chordonomicon_clean.parquet"
    vocab_path = args.data_dir / "vocab.json"
    for path in (parquet_path, vocab_path):
        if not path.exists():
            print(f"Required input not found: {path}", file=sys.stderr)
            print("Run clean_chordonomicon.py first.", file=sys.stderr)
            sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Load vocab + data ---------------------------------------------------
    with open(vocab_path) as f:
        vocab = json.load(f)
    stoi = vocab["token_to_id"]
    itos = {i: t for t, i in stoi.items()}
    vocab_size = len(stoi)

    if vocab_size - 1 > ID_DTYPE_MAX:
        print(f"Vocab size {vocab_size} exceeds uint16 range; widen ID_DTYPE.",
              file=sys.stderr)
        sys.exit(1)

    # Resolve the special token ids we wrap every song with.
    try:
        bos_id = stoi["<bos>"]
        eos_id = stoi["<eos>"]
        unk_id = stoi["<unk>"]
    except KeyError as e:
        print(f"Vocab is missing required special token {e}.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} songs; vocab size {vocab_size:,}")

    # --- Split ---------------------------------------------------------------
    train_df, val_df = stratified_split(df, args.val_frac, args.seed)
    print(f"Split (stratified by genre): "
          f"{len(train_df):,} train / {len(val_df):,} val")

    # Show per-genre split proportions as a sanity check on stratification.
    print("\nPer-genre split:")
    train_counts = train_df["main_genre"].value_counts()
    val_counts = val_df["main_genre"].value_counts()
    for genre in sorted(df["main_genre"].unique()):
        tr = int(train_counts.get(genre, 0))
        va = int(val_counts.get(genre, 0))
        print(f"    {genre:<14} train {tr:>7,}   val {va:>6,}")

    # --- Encode --------------------------------------------------------------
    train_ids, train_unk, train_off, prefix_len = encode_split(
        train_df, stoi, bos_id, eos_id, unk_id)
    val_ids, val_unk, val_off, _ = encode_split(
        val_df, stoi, bos_id, eos_id, unk_id)

    print(f"\nEncoded tokens: {len(train_ids):,} train / {len(val_ids):,} val")
    print(f"Songs: {len(train_off):,} train / {len(val_off):,} val "
          f"(conditioning prefix length = {prefix_len})")
    if train_unk or val_unk:
        # Expected to be zero: the vocab came from this same corpus. Nonzero
        # means a token in the data wasn't in vocab.json (pipeline mismatch).
        print(f"WARNING: fallback-to-<unk> hits: "
              f"{train_unk} train / {val_unk} val (expected 0).")

    # --- Write binaries + meta ----------------------------------------------
    train_ids.tofile(args.outdir / "train.bin")
    val_ids.tofile(args.outdir / "val.bin")
    print(f"\nWrote {args.outdir / 'train.bin'} and {args.outdir / 'val.bin'}")

    # Song start offsets for the v2 document-aware batcher. The v1 (minimal)
    # batcher ignores these, so writing them is harmless to the v1 model.
    np.save(args.outdir / "train.offsets.npy", train_off)
    np.save(args.outdir / "val.offsets.npy", val_off)
    print(f"Wrote train.offsets.npy ({len(train_off):,}) and "
          f"val.offsets.npy ({len(val_off):,})")

    # meta.pkl: nanoGPT reads vocab_size/stoi/itos; prefix_len is extra metadata
    # the v2 batcher reads for prefix masking (ignored by stock nanoGPT).
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos,
            "prefix_len": prefix_len}
    with open(args.outdir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Wrote {args.outdir / 'meta.pkl'} (vocab_size={vocab_size}, "
          f"prefix_len={prefix_len})")

    print("\nDone. Point nanoGPT at this with dataset dir =", args.outdir)


if __name__ == "__main__":
    main()
