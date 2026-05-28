#!/usr/bin/env python3
"""
clean_chordonomicon.py
======================

Prune and reformat the Chordonomicon dataset into a training-ready form for a
conditional chord-progression generator conditioned on (main_genre, decade).

The raw dataset is streamed directly from the Hugging Face Hub
(ailsntua/Chordonomicon) via pandas' fsspec integration, so the large CSV never
needs to live in the git repo. This requires `huggingface_hub` (and `fsspec`,
which it depends on) to be installed so the `hf://` protocol resolves:

    pip install "huggingface_hub[hf_transfer]" fsspec

Pruning / transformation recipe (decided from dataset analysis):
  1. Keep only rows where main_genre AND decade are present.
  2. Drop rows with decade < MIN_DECADE (default 1950) -- pre-1950 buckets are
     tiny (<2k rows total) and contain data-entry noise (e.g. 1890, 1900).
  3. Collapse structure tags to their 8 base types by stripping the trailing
     index:  <verse_2> -> <verse>,  <chorus_11> -> <chorus>, etc.
  4. Build a chord vocabulary from chords occurring >= MIN_CHORD_FREQ times
     (default 100). Rarer chords are replaced with <unk> (or optionally dropped).
  5. Keep only the columns we train on: main_genre, decade, chords.
  6. Serialize each row into a single token string with conditioning prefix:
        <genre:pop> <decade:2010> <verse> C G Amin F ...

Outputs:
  - cleaned dataset (parquet + optional csv)
  - vocab.json  (all special tokens + chord tokens + index mappings)
  - report.txt  (before/after counts, distributions, vocab stats)

Usage:
  python clean_chordonomicon.py --outdir ./cleaned
  python clean_chordonomicon.py --min-decade 1960 --min-chord-freq 50 \
         --rare-chord-policy drop --write-csv
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------------- 
# Configuration / constants
# ----------------------------------------------------------------------------- 

# Source dataset. Streamed straight from the Hugging Face Hub through pandas'
# fsspec integration; nothing is committed to the repo. Override with --dataset
# if you mirror the file elsewhere (any local path or fsspec URL works).
DEFAULT_DATASET = (
    "hf://datasets/ailsntua/Chordonomicon/chordonomicon_v2.csv"
)

# The 8 valid structural section types. Anything else (e.g. the two one-off
# "<intro_riff_N>" tags) is normalized into the closest base type or dropped.
VALID_SECTIONS = {
    "intro", "verse", "chorus", "bridge",
    "outro", "instrumental", "interlude", "solo",
}

# Matches a structure tag like "<verse_12>" or "<intro_riff_2>" and captures the
# leading alphabetic part before the first underscore-number.
TAG_RE = re.compile(r"<([a-zA-Z_]+?)(?:_\d+)?>")

# Special tokens that are always in the vocabulary regardless of the data.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
BASE_SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


# ----------------------------------------------------------------------------- 
# Tag + chord parsing
# ----------------------------------------------------------------------------- 

def normalize_tag(raw_section: str) -> str | None:
    """Map a raw captured section name to a canonical section, or None to drop.

    'verse' -> 'verse'      (already canonical)
    'intro_riff' -> 'intro' (fold the one-off riff tags into intro)
    'chorus' -> 'chorus'
    Anything unrecognized -> None (caller drops it).
    """
    section = raw_section.lower()
    if section in VALID_SECTIONS:
        return section
    # Fold odd compound tags onto their leading base type if we recognize it.
    head = section.split("_", 1)[0]
    if head in VALID_SECTIONS:
        return head
    return None


def tokenize_chords_field(chords: str) -> list[str]:
    """Turn the raw 'chords' string into a flat list of tokens.

    Structure tags are collapsed to canonical '<section>' tokens; chords are
    passed through verbatim (vocabulary filtering happens in a later pass so we
    can compute frequencies across the whole corpus first).
    """
    tokens: list[str] = []
    for piece in chords.strip().split():
        if piece.startswith("<") and piece.endswith(">"):
            m = TAG_RE.fullmatch(piece)
            if not m:
                continue  # malformed tag -> skip
            canonical = normalize_tag(m.group(1))
            if canonical is not None:
                tokens.append(f"<{canonical}>")
            # unrecognized tag -> silently dropped
        else:
            tokens.append(piece)  # a chord token
    return tokens


def is_chord_token(tok: str) -> bool:
    """A token is a chord iff it is not a <...> tag."""
    return not (tok.startswith("<") and tok.endswith(">"))


def genre_to_token(genre: str) -> str:
    """Build a whitespace-free genre token.

    Genres like 'pop rock' contain spaces, but every downstream consumer
    tokenizes sequences with str.split(), which would tear '<genre:pop rock>'
    into two invalid tokens. Replacing spaces with underscores keeps the whole
    token intact through split(). MUST be used everywhere a genre token is
    constructed (vocab build AND row serialization) so the two never diverge.
    """
    return f"<genre:{genre.replace(' ', '_')}>"


# ----------------------------------------------------------------------------- 
# Main cleaning pipeline
# ----------------------------------------------------------------------------- 

def clean(
    input_path: str,
    outdir: Path,
    min_decade: int,
    min_chord_freq: int,
    min_chord_len: int,
    rare_chord_policy: str,
    write_csv: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    log("=" * 70)
    log("Chordonomicon cleaning report")
    log("=" * 70)

    # --- Load ----------------------------------------------------------------
    # Only read the columns we actually use; this keeps memory modest on 680k rows.
    # `input_path` may be a local file or a remote fsspec URL (e.g. hf://...);
    # pandas resolves both transparently.
    usecols = ["main_genre", "decade", "chords"]
    log(f"\nLoading {input_path} (columns: {usecols}) ...")

    # Confirm the expected columns exist before the full read, so a header
    # mismatch fails loudly with a helpful message instead of cryptically.
    header_cols = pd.read_csv(input_path, nrows=0).columns.tolist()
    missing = [c for c in usecols if c not in header_cols]
    if missing:
        log(f"ERROR: expected column(s) {missing} not found in file.")
        log(f"Columns found in file: {header_cols}")
        sys.exit(1)

    # Read every used column as a plain string with the python engine and
    # dtype inference fully disabled. This sidesteps a pandas C-parser bug
    # (IndexError in _concatenate_chunks) that can trigger on large files when
    # usecols is combined with dtypes that vary across read chunks. We parse
    # `decade` to a number ourselves immediately afterwards.
    df = pd.read_csv(
        input_path,
        usecols=usecols,
        dtype=str,
        keep_default_na=True,
        engine="python",
    )
    # `decade` came in as text like "2010" or "2010.0"; coerce to nullable Int.
    df["decade"] = pd.to_numeric(df["decade"], errors="coerce").astype("Int64")

    total_rows = len(df)
    log(f"Raw rows: {total_rows:,}")

    # --- Step 1: require genre AND decade AND chords -------------------------
    df = df.dropna(subset=["main_genre", "decade", "chords"])
    after_required = len(df)
    log(f"\n[1] Rows with main_genre + decade + chords all present: "
        f"{after_required:,}  (dropped {total_rows - after_required:,})")

    # decade is a nullable Int64 after load; rows with NA were just dropped,
    # so it's safe to convert to a plain int column now.
    df["decade"] = df["decade"].astype(int)

    # --- Step 2: drop pre-MIN_DECADE rows ------------------------------------
    before_decade = len(df)
    df = df[df["decade"] >= min_decade]
    after_decade = len(df)
    log(f"[2] Rows with decade >= {min_decade}: {after_decade:,}  "
        f"(dropped {before_decade - after_decade:,})")

    # Normalize genre whitespace/case lightly (values are already clean classes).
    df["main_genre"] = df["main_genre"].str.strip()

    # --- Step 3: tokenize + collapse structure tags --------------------------
    log("\n[3] Tokenizing chord fields and collapsing structure tags ...")
    df["tokens"] = df["chords"].map(tokenize_chords_field)

    # Drop any rows that ended up with no actual chord tokens (defensive).
    has_chord = df["tokens"].map(lambda ts: any(is_chord_token(t) for t in ts))
    before_empty = len(df)
    df = df[has_chord]
    log(f"    Rows remaining after removing empty/tag-only progressions: "
        f"{len(df):,}  (dropped {before_empty - len(df):,})")

    # --- Step 4: build chord vocabulary by frequency -------------------------
    log("\n[4] Building chord vocabulary ...")
    chord_counts: Counter[str] = Counter()
    for toks in df["tokens"]:
        for t in toks:
            if is_chord_token(t):
                chord_counts[t] += 1

    distinct_chords = len(chord_counts)
    kept_chords = sorted(
        [c for c, n in chord_counts.items() if n >= min_chord_freq]
    )
    rare_chords = {c for c, n in chord_counts.items() if n < min_chord_freq}
    kept_mass = sum(n for c, n in chord_counts.items() if n >= min_chord_freq)
    total_mass = sum(chord_counts.values())

    log(f"    Distinct chords seen: {distinct_chords:,}")
    log(f"    Chords kept (freq >= {min_chord_freq}): {len(kept_chords):,}")
    log(f"    Chords treated as rare: {len(rare_chords):,}")
    log(f"    Token coverage of kept chords: "
        f"{kept_mass / total_mass:.4%} of all chord tokens")
    log(f"    Rare-chord policy: {rare_chord_policy}")

    kept_chord_set = set(kept_chords)

    def apply_rare_policy(toks: list[str]) -> list[str]:
        out: list[str] = []
        for t in toks:
            if is_chord_token(t) and t not in kept_chord_set:
                if rare_chord_policy == "unk":
                    out.append(UNK_TOKEN)
                # 'drop' -> skip the token entirely
            else:
                out.append(t)
        return out

    df["tokens"] = df["tokens"].map(apply_rare_policy)

    # If dropping rare chords emptied a progression, remove that row too.
    if rare_chord_policy == "drop":
        has_chord = df["tokens"].map(
            lambda ts: any(is_chord_token(t) for t in ts)
        )
        before = len(df)
        df = df[has_chord]
        if before != len(df):
            log(f"    Rows dropped after rare-chord removal left them empty: "
                f"{before - len(df):,}")

    # --- Step 4b: minimum chord-length filter --------------------------------
    # Count only actual chord tokens (section tags don't count toward length),
    # measured AFTER the rare-chord policy so that dropping rare chords can push
    # a borderline progression below the floor. Sequences shorter than this are
    # too short to contain a meaningful progression (a length-1 "song" is just a
    # single chord) and are pure noise once concatenated for training.
    def chord_count(toks: list[str]) -> int:
        return sum(1 for t in toks if is_chord_token(t))

    df["n_chords"] = df["tokens"].map(chord_count)
    before_len = len(df)
    df = df[df["n_chords"] >= min_chord_len]
    log(f"\n[4b] Rows with >= {min_chord_len} chord tokens: {len(df):,}  "
        f"(dropped {before_len - len(df):,})")

    # --- Step 5: build full vocabulary + conditioning tokens -----------------
    genres = sorted(df["main_genre"].unique())
    decades = sorted(int(d) for d in df["decade"].unique())
    genre_tokens = [genre_to_token(g) for g in genres]
    decade_tokens = [f"<decade:{d}>" for d in decades]
    section_tokens = [f"<{s}>" for s in sorted(VALID_SECTIONS)]

    vocab_list = (
        BASE_SPECIAL_TOKENS
        + genre_tokens
        + decade_tokens
        + section_tokens
        + kept_chords
    )
    # Guard against accidental duplicates while preserving order.
    seen: set[str] = set()
    vocab_list = [t for t in vocab_list if not (t in seen or seen.add(t))]
    token_to_id = {tok: i for i, tok in enumerate(vocab_list)}

    log("\n[5] Vocabulary assembled:")
    log(f"    Special tokens : {len(BASE_SPECIAL_TOKENS)}")
    log(f"    Genre tokens   : {len(genre_tokens)}  -> {genres}")
    log(f"    Decade tokens  : {len(decade_tokens)} -> {decades}")
    log(f"    Section tokens : {len(section_tokens)}")
    log(f"    Chord tokens   : {len(kept_chords)}")
    log(f"    TOTAL vocab    : {len(vocab_list):,}")

    # --- Step 6: serialize each row into a training string -------------------
    def serialize(row) -> str:
        prefix = [genre_to_token(row["main_genre"]), f"<decade:{row['decade']}>"]
        return " ".join(prefix + row["tokens"])

    df["sequence"] = df.apply(serialize, axis=1)
    df["seq_len"] = df["tokens"].map(len)  # length excluding the 2 prefix tokens

    # --- Distributions for the report ----------------------------------------
    log("\nFinal main_genre distribution:")
    for g, n in df["main_genre"].value_counts().items():
        log(f"    {g:<14} {n:>8,}")

    log("\nFinal decade distribution:")
    for d, n in df["decade"].value_counts().sort_index().items():
        log(f"    {d:<6} {n:>8,}")

    log("\nProgression length (tokens, excl. conditioning prefix):")
    log(f"    min   : {df['seq_len'].min():,}")
    log(f"    median: {int(df['seq_len'].median()):,}")
    log(f"    mean  : {df['seq_len'].mean():.1f}")
    log(f"    p95   : {int(df['seq_len'].quantile(0.95)):,}")
    log(f"    max   : {df['seq_len'].max():,}")

    log(f"\nFINAL ROW COUNT: {len(df):,}  "
        f"({len(df) / total_rows:.1%} of the original {total_rows:,})")

    # --- Write outputs -------------------------------------------------------
    out_cols = ["main_genre", "decade", "sequence", "seq_len"]

    # Prefer parquet (compact, preserves dtypes), but fall back to CSV if no
    # parquet engine (pyarrow/fastparquet) is installed so a full run is never
    # lost over a missing optional dependency.
    wrote_parquet = False
    try:
        parquet_path = outdir / "chordonomicon_clean.parquet"
        df[out_cols].to_parquet(parquet_path, index=False)
        wrote_parquet = True
        log(f"\nWrote dataset -> {parquet_path}")
    except ImportError:
        log("\nNOTE: no parquet engine (pyarrow/fastparquet) found; "
            "writing CSV instead. Install pyarrow for compact parquet output.")

    if write_csv or not wrote_parquet:
        csv_path = outdir / "chordonomicon_clean.csv"
        df[out_cols].to_csv(csv_path, index=False)
        log(f"Wrote dataset -> {csv_path}")

    vocab_path = outdir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(
            {
                "token_to_id": token_to_id,
                "id_to_token": vocab_list,
                "special_tokens": BASE_SPECIAL_TOKENS,
                "genre_tokens": genre_tokens,
                "decade_tokens": decade_tokens,
                "section_tokens": section_tokens,
                "config": {
                    "dataset": input_path,
                    "min_decade": min_decade,
                    "min_chord_freq": min_chord_freq,
                    "min_chord_len": min_chord_len,
                    "rare_chord_policy": rare_chord_policy,
                },
            },
            f,
            indent=2,
        )
    log(f"Wrote vocabulary -> {vocab_path}")

    report_path = outdir / "report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    log(f"Wrote report -> {report_path}")


# ----------------------------------------------------------------------------- 
# CLI
# ----------------------------------------------------------------------------- 

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Clean the Chordonomicon dataset.")
    p.add_argument("--dataset", default=DEFAULT_DATASET, type=str,
                   help="Source dataset path. Defaults to streaming the raw CSV "
                        "from the Hugging Face Hub; can be overridden with any "
                        "local path or fsspec URL.")
    p.add_argument("--outdir", default=Path("./cleaned"), type=Path,
                   help="Directory for cleaned outputs (default: ./cleaned).")
    p.add_argument("--min-decade", default=1950, type=int,
                   help="Drop rows with decade earlier than this (default 1950).")
    p.add_argument("--min-chord-freq", default=100, type=int,
                   help="Minimum corpus frequency for a chord to enter the "
                        "vocabulary (default 100).")
    p.add_argument("--min-chord-len", default=8, type=int,
                   help="Drop progressions with fewer than this many chord "
                        "tokens, counted after the rare-chord policy is applied "
                        "and excluding section tags (default 8).")
    p.add_argument("--rare-chord-policy", choices=["unk", "drop"], default="unk",
                   help="What to do with sub-threshold chords (default unk).")
    p.add_argument("--write-csv", action="store_true",
                   help="Also write a CSV copy alongside the parquet.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    clean(
        input_path=args.dataset,
        outdir=args.outdir,
        min_decade=args.min_decade,
        min_chord_freq=args.min_chord_freq,
        min_chord_len=args.min_chord_len,
        rare_chord_policy=args.rare_chord_policy,
        write_csv=args.write_csv,
    )


if __name__ == "__main__":
    main()
