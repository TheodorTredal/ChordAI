#!/usr/bin/env bash
cd "$(dirname "$0")/nanoGPT" || exit 1

# Generate from genre + decade alone
python3 sample.py --out_dir=out-chords --mode=generate --genre=pop --decade=2010 --device=cuda --temperature=1.0

# Extend a seed progression
python3 sample.py --out_dir=out-chords --mode=extend \
    --genre=rock --decade=1990 --seed_chords="Emin C G D" --device=cuda --temperature=1.1

# Verse -> chorus
python3 sample.py --out_dir=out-chords --mode=section \
    --genre=pop --decade=2010 \
    --seed_chords="<verse> C G Amin F C G Amin F" --next_section=chorus --device=cuda --temperature=1.0
