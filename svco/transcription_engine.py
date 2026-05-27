"""Local speech/chord parsing and payload assembly."""

from __future__ import annotations

import tempfile
import wave
from typing import List, TypedDict, Any
import numpy as np
import soundfile as sf
import os

SAMPLE_RATE = 16_000
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class UserTurnPayload(TypedDict):
    user_speech: str
    user_chords: List[Any]
    user_instruction: str
def transcribe_chords_offline_from_pcm(
    chord_bytes: bytes,
    *,
    pcm_sr: int,
    pcm_channels: int = 1,
    sample_width_bytes: int = 2,  # int16
    hop_s: float = 0.10,
    min_seg_s: float = 0.50,
    change_cost: float = 2.0,
    rms_threshold: float = 0.01,
):
    """
    Convert raw PCM int16 audio bytes to a temp WAV and run offline chord transcription.
    Returns the offline_chords JSON dict (with segments).
    """
    if not chord_bytes:
        return {"segments": [], "params": {}, "sr": None}

    import tempfile
    import wave
    import numpy as np

    # Lazy import to avoid import-time failures in environments without librosa deps
    from svco.offline_chords import transcribe_file

    # Validate buffer length
    if len(chord_bytes) < sample_width_bytes * pcm_channels:
        return {"segments": [], "params": {}, "sr": None}

    # Write WAV (PCM_16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(pcm_channels)
            wf.setsampwidth(sample_width_bytes)
            wf.setframerate(pcm_sr)
            wf.writeframes(chord_bytes)

        result = transcribe_file(
            wav_path,
            target_sr=22050,          # analysis SR
            hop_s=hop_s,
            min_seg_s=min_seg_s,
            change_cost=change_cost,
            rms_threshold=rms_threshold,
        )
        return result
    finally:
        import os
        try:
            os.remove(wav_path)
        except OSError:
            pass

def transcribe_voice(speech_bytes: bytes) -> str:
    """Transcribe speech bytes with local pywhispercpp bindings."""
    if not speech_bytes:
        return ""

    try:
        from pywhispercpp.model import Model
    except ImportError:
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
        with wave.open(temp_wav.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(speech_bytes)

        model = Model("base.en")
        result = model.transcribe(temp_wav.name)

    if isinstance(result, str):
        return result.strip()

    if isinstance(result, list):
        chunks = [str(item.get("text", "")).strip() for item in result if isinstance(item, dict)]
        return " ".join(chunk for chunk in chunks if chunk).strip()

    return ""

def transcribe(speech_bytes, chord_bytes):
    import whisper
    print("[transcription_engine] Transcribing buffers...")

    # --- Speech recognition using Whisper ---
    user_speech = ""
    if speech_bytes and len(speech_bytes) > 0:
        # Convert buffer to temp WAV for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Decode raw PCM and write as WAV
            audio = np.frombuffer(speech_bytes, dtype=np.int16)
            sf.write(tmp, audio, samplerate=16000, subtype='PCM_16')
            tmp_path = tmp.name
        
        try:
            model = whisper.load_model("base")  # You can use "tiny" or "small" for speed
            result = model.transcribe(tmp_path, fp16=False)
            user_speech = result.get('text', '').strip()
        finally:
            os.remove(tmp_path)
    else:
        user_speech = "(no speech detected)"

    # --- Chord recognition [stub, implement later] ---
    # Real chord recognition is research-level; stub returns a dummy
    user_chords = []
    if chord_bytes and len(chord_bytes) > 0:
        # TODO: Replace this dummy with a real system if desired
        user_chords = ["C", "G", "Am", "F"]
    else:
        user_chords = []

    # You could apply some logic to extract an "instruction" from the transcript.
    user_instruction = user_speech if user_speech else ""

    return {
        "user_speech": user_speech,
        "user_chords": user_chords,
        "user_instruction": user_instruction
    }
def transcribe_chords(chord_bytes: bytes) -> List[str]:
    """Mock lightweight chord identification based on chroma peak strength."""
    if not chord_bytes:
        return []

    try:
        import librosa
        import numpy as np
    except ImportError:
        return []

    audio = np.frombuffer(chord_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return []

    chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
    mean_chroma = chroma.mean(axis=1)
    if mean_chroma.size != 12:
        return []

    top_indices = np.argsort(mean_chroma)[-4:][::-1]
    return [f"{PITCH_CLASSES[int(idx)]}maj" for idx in top_indices]


def build_payload(user_speech: str, user_chords: Any, user_instruction: str) -> UserTurnPayload:
    return {
        "user_speech": user_speech,
        "user_chords": user_chords,
        "user_instruction": user_instruction,
    }


def process_turn(speech_bytes: bytes, chord_bytes: bytes, user_instruction: str) -> UserTurnPayload:
    speech_text = transcribe_voice(speech_bytes)

    # IMPORTANT: set this to whatever your stream_ingest actually uses for chord audio
    # If your chord buffer is recorded at 16000 in stream_ingest, keep 16000.
    CHORD_PCM_SR = SAMPLE_RATE  # replace with config if you have one

    chord_result = transcribe_chords_offline_from_pcm(
        chord_bytes,
        pcm_sr=CHORD_PCM_SR,
        hop_s=0.10,
        min_seg_s=0.50,
        change_cost=2.0,
        rms_threshold=0.01,
    )

    # You can send full segments...
    chord_segments = chord_result.get("segments", [])

    return build_payload(
        user_speech=speech_text,
        user_chords=chord_segments,
        user_instruction=user_instruction,
    )