#!/usr/bin/env bash
cd "$(dirname "$0")/nanoGPT" || exit 1
python train.py config/train_chords.py --doc_aware_batching=True # When running on a local mac: --device='mps' --compile=False
