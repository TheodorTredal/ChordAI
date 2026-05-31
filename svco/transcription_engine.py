"""Local chord parsing and payload assembly."""

from __future__ import annotations

import tempfile
import wave
import os

SAMPLE_RATE = 16000  # MUST match stream_ingest capture rate


def transcribe_chords_offline_from_pcm(
    chord_bytes: bytes,
    *,
    pcm_sr: int,
    pcm_channels: int = 1,
    sample_width_bytes: int = 2,  # int16
    hop_s: float = 0.10,
    min_seg_s: float = 0.50,
    change_cost: float = 2.0,
    rms_threshold: float = 0.005,  
):
    """
    Convert raw PCM int16 audio bytes to a temp WAV and run offline Madmom chord transcription.

    Returns:
      dict with keys like {"segments": [...], "sr": ..., "params": ...}
    """
    min_sec = 1.0
    min_bytes = int(min_sec * pcm_sr * sample_width_bytes * pcm_channels)
    
    if len(chord_bytes) < min_bytes:
        return {"segments": [], "params": {"skipped": "too_short", "seconds": len(chord_bytes) / (pcm_sr*sample_width_bytes*pcm_channels)}, "sr": pcm_sr}
    
    if not chord_bytes:
        return {"segments": [], "params": {}, "sr": None}

    # Validate buffer length
    if len(chord_bytes) < sample_width_bytes * pcm_channels:
        return {"segments": [], "params": {}, "sr": None}

    # Lazy import pointing to the retained Madmom engine
    from svco.madmom_chords import transcribe_file

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

