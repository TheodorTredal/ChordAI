# Sequential Voice-and-Chord Orchestrator (SVCO)

> Dual-channel audio ingestion pipeline for transcribing speech and acoustic guitar chords to interface with LLM agents via the Model Context Protocol (MCP).

## Overview

SVCO is a local, Python-based orchestration application designed to capture audio from a musician alternating between spoken instructions and playing guitar chords. The system routes the audio, transcribes speech using Whisper, extracts chords using deep learning (Madmom) or CQT-based template matching, and formats the harmonic timeline and speech into a structured payload. This payload is then passed to Claude (via Anthropic API) to interact with backend generators via MCP.

The architecture supports both **live microphone streaming** (optimized for WSLg) and **offline file processing** (interfacing with web frontends via FastAPI).

---

## 🛠️ System Prerequisites

Before configuring the Python environment, ensure your host operating system has the necessary binary dependencies installed. This is strictly required for transcoding and capturing audio in Linux/WSL environments.

```bash
# Update package lists
sudo apt update

# Install ffmpeg (required for transcoding compressed browser formats like .webm or .m4a)
# Install pulseaudio-utils (required for `parec` live audio capture in WSLg)
sudo apt install -y ffmpeg pulseaudio-utils

```


## Note on WSL Environments
If you are running this in the Windows Subsystem for Linux (WSL), be aware that standard PyAudio/ALSA implementations will likely fail to detect the host microphone. This project uses parec via PulseAudio to bypass this limitation. Ensure your Windows privacy settings allow WSL to access your microphone.

## 📦 Installation
Due to the strict compilation requirements of the deep learning Music Information Retrieval (MIR) modules, the Python dependencies must be installed in stages. Attempting to install them all at once will cause the Cython compilation to fail.

Requires Python 3.10+.

```bash
# 1. Install foundational build tools and audio libraries
pip install numpy cython torch librosa soundfile

# 2. Install the Deep Learning chord extraction framework
# (Must be installed after Cython and NumPy)
pip install mido
pip install git+[https://github.com/CPJKU/madmom.git](https://github.com/CPJKU/madmom.git)

# 3. Install orchestration, synthesis, and web framework tools
pip install anthropic mcp kokoro sounddevice fastapi uvicorn python-multipart
```

## 🏗️ Project Architecture
The codebase is strictly separated by concern, ensuring that live microphone ingestion does not block heavy offline transcription math or web server routing.

Core Processing (svco/)
- pulse_ingest.py: Captures live audio streams using a parec subprocess. Utilizes Silero VAD and smoothed Exponential Moving Average (EMA) RMS thresholds to securely segment audio into speech and chord turns.

- offline_processor.py: Accepts pre-recorded audio files, managing resampling and transcoding into the strict 16kHz mono PCM bytes required by the transcription engine.

- transcription_engine.py: Coordinates the transcription of voice data and the formatting of offline chord data.

- madmom_chords.py: Employs a Convolutional Neural Network (CNN) and Conditional Random Field (CRF) for high-accuracy offline chord transcription, immune to the acoustic flickering of standard template matching.

- offline_chords.py: A fast, baseline acoustic chord recognizer utilizing Constant-Q Transform (CQT) chroma templates, Viterbi decoding, and a low-frequency bass-bias.

- agent_orchestrator.py: Formats the transcribed speech and smoothed chord timeline into a structured JSON payload, passing it to Claude via MCP.

- output_synthesis.py: Handles UI data updates and local Text-to-Speech (TTS) response generation.

- main.py: The pipeline orchestrator for live microphone streaming mode.

API Layer (server/)
- api.py: A FastAPI application that serves as the entry point for web frontend clients (e.g., Nuxt). Receives FormData audio uploads via a /verify-chords endpoint, allowing users to bypass noisy live microphones.

## 🚀 Usage Guide
1. Live Microphone Mode
To run the continuous pipeline that listens for alternating speech and chords, run the main orchestrator. Make sure you stay silent for the configured timeout (e.g., ~2.5 seconds) to trigger the buffer flush.

```Bash
python -m svco.main
```

2. Local File Testing Mode
To bypass the microphone and test the chord transcription engines directly with a pre-recorded file (.wav, .mp3, .m4a):

```Bash
python test_local.py path/to/your_audio_file.wav
```

(Note: test_local.py will output both the legacy CQT template results and the Madmom CNN results so you can compare accuracy).

3. Web API Server Mode
To boot the FastAPI server to receive audio uploads from a frontend web application:

```Bash
# Run from the root directory
uvicorn server.api:app --reload
```
The server will boot at http://127.0.0.1:8000. The frontend can send POST requests containing audio files to the /verify-chords endpoint.

## 🎛️ Configuration & Tuning
If the application is struggling to trigger chord recordings (or is getting stuck buffering indefinitely), adjust the hardware thresholds in svco/pulse_ingest.py:

- CHORD_START_RMS_THRESHOLD: The volume level required to open the chord buffer. Lower this if quiet acoustic guitar strums are being ignored.

- CHORD_KEEP_RMS_THRESHOLD: The volume level required to maintain an open buffer.

- CHORD_HANGOVER_S: How long (in seconds) to keep recording after the volume dips below the threshold (prevents chord clipping between strums).
