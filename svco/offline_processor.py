"""Feature to process pre-recorded speech and chord audio files."""

from __future__ import annotations

import logging
import librosa
import numpy as np

from svco.transcription_engine import process_turn
from svco.agent_orchestrator import orchestrate
from svco.output_synthesis import synthesize_output

# Must match transcription_engine.py and stream_ingest.py
TARGET_SAMPLE_RATE = 16000 

def _load_audio_to_pcm_bytes(file_path: str, target_sr: int) -> bytes:
    """
    Loads an audio file (MP3/WAV), resamples it to the target rate,
    and converts it to raw int16 PCM bytes.
    """
    try:
        # librosa handles format decoding, downmixing to mono, and resampling automatically
        audio_float, _ = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Convert float32 [-1.0, 1.0] array to int16 [-32768, 32767] array
        audio_int16 = (audio_float * 32768.0).astype(np.int16)
        
        return audio_int16.tobytes()
    except Exception as e:
        logging.error(f"Failed to load audio file {file_path}: {e}")
        return b""

def process_files(speech_file_path: str, chord_file_path: str) -> dict:
    """
    Ingests two distinct audio files, builds the payload, and orchestrates the agent response.
    """
    logging.info(f"Loading speech audio from: {speech_file_path}")
    speech_bytes = _load_audio_to_pcm_bytes(speech_file_path, TARGET_SAMPLE_RATE)
    
    logging.info(f"Loading chord audio from: {chord_file_path}")
    chord_bytes = _load_audio_to_pcm_bytes(chord_file_path, TARGET_SAMPLE_RATE)

    if not speech_bytes and not chord_bytes:
        raise ValueError("Both audio buffers are empty. Check file paths and formats.")

    logging.info("Transcribing audio into turn payload...")
    # Generate the payload using your existing engine
    payload = process_turn(speech_bytes, chord_bytes, user_instruction="")
    
    logging.info(f"Generated Payload: {payload}")
    
    logging.info("Sending payload to orchestrator...")
    # Retrieve the final response from Claude (or your local agent)
    response_json = orchestrate(payload)
    
    return response_json

if __name__ == "__main__":
    # Example usage for testing locally
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 3:
        print("Usage: python -m svco.offline_processor <speech.wav/mp3> <chords.wav/mp3>")
        sys.exit(1)
        
    speech_path = sys.argv[1]
    chord_path = sys.argv[2]
    
    final_answer = process_files(speech_path, chord_path)
    
    print("\n=== Final Response ===")
    synthesize_output(final_answer)