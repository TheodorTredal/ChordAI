# Sequential Voice-and-Chord Orchestrator

> Speech and chords to text — text and chords to speech

## Overview

This project captures microphone input, splits and classifies it into speech and guitar chords, transcribes speech to text, optionally detects chords, and then communicates with Claude (AI) and a generator module through MCP for creative music interactions.

## Quickstart

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

*You need Python 3.8+ with pip. PyAudio may require compilation tools and `portaudio19-dev` on Linux:*

```sh
sudo apt-get install portaudio19-dev python3-dev
```

### 2. Run the main pipeline

```sh
python main.py
```

- Speak and/or play chords into your default microphone.
- After a period of silence, the input is processed and sent through the pipeline.
- Output is displayed at the end of each "turn."
- Currently, speech-to-text uses OpenAI Whisper, and chord recognition is a stub (to be improved).

---

## Architecture

- **stream_ingest.py:** Captures and routes mic input into speech/chord buffers using VAD and RMS.
- **transcription_engine.py:** Uses Whisper to transcribe speech and (soon) Chordino or chroma features for chord recognition.
- **agent_orchestrator.py:** Sends text/chords to Claude (and generator via MCP).
- **output_synthesis.py:** Displays or renders output.
- **main.py:** Pipeline orchestrator.

---

## Advanced (Improving Chord Recognition)

For higher chord accuracy, install **Sonic Annotator** and Chordino Vamp plugin (Linux/Mac):
```sh
sudo apt install vamp-plugin-sdk sonic-annotator wget unzip
wget https://www.isophonics.net/files/chordino-vamp.1.1.linux64.zip
unzip chordino-vamp.1.1.linux64.zip
sudo cp chordino-vamp.1.1.linux64/libvamp-chordino.so /usr/lib/vamp/
```
See [project documentation](https://www.isophonics.net/nnls-chroma) for details.

---

## OS Note

- If using WSL, mic support may be limited and audio device errors may occur.
- For real-time music/chord analysis, prefer Linux (native/VM) or Mac.

---

## Status

- [x] Speech detection (VAD)
- [x] Speech transcription (Whisper)
- [ ] Real chord recognition (integrate Chordino or a simple classifier)
- [x] Claude/MCP integration stubbed
- [ ] End-to-end music/text creative exchange

---

## License

MIT (or your project’s chosen license)
