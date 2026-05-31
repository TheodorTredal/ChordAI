## SVCO (Chord Transcription Engine)

The Python-based SVCO module acts as the core acoustic analysis engine for the ChordAI platform. Following the deprecation of the live voice ingestion pipeline, this root directory is strictly dedicated to ingesting audio files and extracting timestamped harmonic progressions. The remnants of the scrapped live-voice interaction idea have been retained as proof of concept and archived entirely within the legacy_voice_agent_v1 folder.

## Architecture & Codebase Layout
The directory is structured to separate the audio preparation logic from the heavy mathematical and deep learning models, while keeping deprecated concepts isolated.

Plaintext
```

svco/
├── config.py                 # Core constants including target sample rates and buffer sizes
├── transcription_engine.py   # PCM data validation, temporary file generation, and engine routing
├── madmom_chords.py          # Primary deep learning orchestration runner
├── offline_chords.py         # Secondary mathematical baseline chord recognizer
├── offline_processor.py      # Pre-recorded audio loading and resampling wrapper
└── legacy_voice_agent_v1/    # Proof-of-concept archive containing the remnants of the scrapped live voice agent
```

## Core Infrastructure Responsibilities
### 1. Audio Data Bridging

The transcription_engine.py script acts as the entry point for the underlying chord recognition models. It accepts raw uncompressed PCM bytes, validates their length, and writes them to temporary WAV files. It ensures that no computing power is wasted on corrupted or insufficiently long audio captures before passing the data to the models.

### 2. Primary Deep Learning Extraction

The madmom_chords.py module houses the chord transcription. It feeds the audio into a Convolutional Neural Network to extract raw chord probabilities. It then applies a Conditional Random Field to smooth these probabilities into definitive, sequential chord segments. This method is highly resistant to acoustic noise and performance artifacts.

### 3. Baseline Acoustic Recognition

The offline_chords.py script provides a fast, mathematically driven alternative to the deep learning model. It utilizes a Constant-Q Transform to create chroma templates and matches the audio against a known vocabulary of major, minor, and suspended chord structures. It relies on Viterbi Decoding to enforce temporal smoothing and structure.

### 4. Legacy Archive (Proof of Concept)

The legacy_voice_agent_v1/ directory serves as the historical proof for the original orchestration idea. It contains the scrapped logic for live microphone streaming, voice-activity detection (pulse_ingest.py), and Anthropic model formatting (agent_orchestrator.py). These components are entirely decoupled from the active transcription pipeline.

## Standard Run Command
To bypass any web servers and test the active transcription engines directly with a pre-recorded audio file (WAV, FLAC, or MP3), navigate to this directory and utilize the baseline recognizer's command-line interface:

```Bash
python -m svco.offline_chords path/to/your_audio_file.wav --out chords.json
```
