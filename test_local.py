"""Local test script to verify offline chord transcription with audio files."""

import sys
import os
import librosa
import numpy as np
from svco.transcription_engine import transcribe_chords_offline_from_pcm
from svco.madmom_chords import transcribe_chords_madmom

TARGET_SAMPLE_RATE = 16000 

def test_legacy_template_engine(file_path: str):
    print("\n--- Running Legacy Template Engine ---")
    audio_float, _ = librosa.load(file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    audio_int16 = (audio_float * 32768.0).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    chord_result = transcribe_chords_offline_from_pcm(
        pcm_bytes,
        pcm_sr=TARGET_SAMPLE_RATE,
        hop_s=0.10,
        min_seg_s=0.50,
        change_cost=2.0,
        rms_threshold=0.005,
    )

    display_chords = [seg.get("display", "N") for seg in chord_result.get("segments", [])]
    filtered_chords = [c for i, c in enumerate(display_chords) if c != "N" and (i == 0 or c != display_chords[i-1])]
    
    print(f"Legacy Result: {filtered_chords}")

def test_madmom_engine(file_path: str):
    print("\n--- Running Madmom Deep Learning Engine ---")
    # Madmom handles the file loading and resampling natively
    result = transcribe_chords_madmom(file_path)
    print(f"Madmom Result: {result}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_local.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: Could not find file at {audio_file}")
        sys.exit(1)
        
    print(f"Loading audio file: {audio_file}")
    
    # Run both to compare the outputs
    test_legacy_template_engine(audio_file)
    test_madmom_engine(audio_file)
    print("\nTest complete.")