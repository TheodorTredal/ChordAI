"""Local speech/chord parsing and payload assembly."""

from __future__ import annotations

import tempfile
import wave
from typing import List, TypedDict, Any
import numpy as np
import soundfile as sf
import os

SAMPLE_RATE = 16000  # MUST match stream_ingest capture rate
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
    rms_threshold: float = 0.005,  # CHANGE THIS: Lowered from 0.01 to match pulse_ingest
):
    """
    Convert raw PCM int16 audio bytes to a temp WAV and run offline chord transcription.

    Returns:
      dict with keys like {"segments": [...], "sr": ..., "params": ...}
      (whatever svco.offline_chords.transcribe_file returns)
    """
    min_sec = 1.0
    min_bytes = int(min_sec * pcm_sr * sample_width_bytes * pcm_channels)
    if len(chord_bytes) < min_bytes:
        return {"segments": [], "params": {"skipped": "too_short", "seconds": len(chord_bytes) / (pcm_sr*sample_width_bytes*pcm_channels)}, "sr": pcm_sr}
    if not chord_bytes:
        return {"segments": [], "params": {}, "sr": None}

    # Lazy import so importing transcription_engine doesn't require librosa at import-time
    from svco.offline_chords import transcribe_file

    # Validate buffer length
    if len(chord_bytes) < sample_width_bytes * pcm_channels:
        return {"segments": [], "params": {}, "sr": None}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(pcm_channels)
            wf.setsampwidth(sample_width_bytes)
            wf.setframerate(pcm_sr)
            wf.writeframes(chord_bytes)

        return transcribe_file(
            wav_path,
            target_sr=22050,  # analysis SR
            hop_s=hop_s,
            min_seg_s=min_seg_s,
            change_cost=change_cost,
            rms_threshold=rms_threshold,
        )
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass



def transcribe_voice(speech_bytes: bytes) -> str:
    """Transcribe speech bytes using the best available local backend.

    Preference order:
      1) pywhispercpp (fast, simple deployment if installed)
      2) openai-whisper (Python package 'whisper')
    """
    if not speech_bytes:
        return ""

    import logging
    import tempfile
    import wave

    # Write WAV (16k, mono, int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav_path = temp_wav.name

    try:
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(speech_bytes)

        # 1) Try pywhispercpp
        try:
            from pywhispercpp.model import Model  # type: ignore
            try:
                model = Model("base.en")
                result = model.transcribe(wav_path)

                if isinstance(result, str):
                    return result.strip()

                if isinstance(result, list):
                    chunks = [str(item.get("text", "")).strip() for item in result if isinstance(item, dict)]
                    return " ".join(c for c in chunks if c).strip()

                return ""
            except Exception as e:
                logging.exception(f"pywhispercpp transcription failed: {e}")
        except ImportError:
            pass

        # 2) Try openai-whisper
        try:
            import whisper  # type: ignore
            try:
                model = whisper.load_model("base")
                out = model.transcribe(wav_path, fp16=False)
                return str(out.get("text", "")).strip()
            except Exception as e:
                logging.exception(f"openai-whisper transcription failed: {e}")
                return ""
        except ImportError:
            logging.warning("No speech transcription backend installed (pywhispercpp or openai-whisper).")
            return "(no STT backend installed)"

    finally:
        import os
        try:
            os.remove(wav_path)
        except OSError:
            pass


def transcribe(speech_bytes: bytes, chord_bytes: bytes):
    """
    Legacy entrypoint used by some code paths.

    Produces:
      - user_speech: Whisper transcript (if whisper is installed) OR a fallback string
      - user_chords: offline chord segments (list[dict])
      - user_instruction: currently same as user_speech
    """
    print("[transcription_engine] Transcribing buffers...")

    # --- Speech recognition using Whisper (optional) ---
    user_speech = ""
    if speech_bytes and len(speech_bytes) > 0:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio = np.frombuffer(speech_bytes, dtype=np.int16)
            sf.write(tmp, audio, samplerate=SAMPLE_RATE, subtype="PCM_16")
            tmp_path = tmp.name

        try:
            try:
                import whisper  # provided by openai-whisper
            except ImportError:
                whisper = None

            if whisper is None:
                user_speech = "(whisper not installed)"
            else:
                model = whisper.load_model("base")  # tiny/small for speed
                result = model.transcribe(tmp_path, fp16=False)
                user_speech = result.get("text", "").strip()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        user_speech = "(no speech detected)"

    # --- Chord recognition (offline segments) ---
    if chord_bytes and len(chord_bytes) > 0:
        chord_result = transcribe_chords_offline_from_pcm(
            chord_bytes,
            pcm_sr=SAMPLE_RATE,  # MUST match stream_ingest capture rate
            hop_s=0.10,
            min_seg_s=0.50,
            change_cost=2.0,
            rms_threshold=0.005,
        )
        user_chords = chord_result.get("segments", [])
    else:
        user_chords = []

    user_instruction = user_speech if user_speech else ""

    return {
        "user_speech": user_speech,
        "user_chords": user_chords,
        "user_instruction": user_instruction,
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

    if speech_bytes and not speech_text.strip():
        # This is critical: makes failures visible in your logs/UI
        speech_text = "(speech captured but transcription empty; check STT backend/logs)"

    chord_result = transcribe_chords_offline_from_pcm(
        chord_bytes,
        pcm_sr=SAMPLE_RATE,
        hop_s=0.10,
        min_seg_s=0.50,
        change_cost=2.0,
        rms_threshold=0.005,
    )
    chord_segments = chord_result.get("segments", [])

    return build_payload(
        user_speech=speech_text,
        user_chords=chord_segments,
        user_instruction=user_instruction,
    )