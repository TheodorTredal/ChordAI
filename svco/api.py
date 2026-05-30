from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import logging
import librosa
import numpy as np
from svco.transcription_engine import transcribe_chords_offline_from_pcm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["POST"],
)

TARGET_SAMPLE_RATE = 16000

@app.post("/verify-chords")
async def verify_chords_endpoint(audio_file: UploadFile = File(...)):
    temp_path = f"temp_{audio_file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        audio_float, _ = librosa.load(temp_path, sr=TARGET_SAMPLE_RATE, mono=True)
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
        
        filtered_chords = []
        for chord in display_chords:
            if chord != "N" and (not filtered_chords or filtered_chords[-1] != chord):
                filtered_chords.append(chord)

        return {"status": "success", "temporary_chords": filtered_chords}

    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)