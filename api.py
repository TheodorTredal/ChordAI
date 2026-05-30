from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import subprocess
import tempfile
import logging
from svco.madmom_chords import transcribe_chords_madmom

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_SAMPLE_RATE = 16000

@app.post("/verify-chords")
async def verify_chords_endpoint(audio_file: UploadFile = File(...)):
    suffix = os.path.splitext(audio_file.filename or "recording.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        shutil.copyfileobj(audio_file.file, f)
        raw_path = f.name

    wav_path = raw_path + ".wav"

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path,
             "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", wav_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg: {result.stderr.decode()}")

        filtered_chords = transcribe_chords_madmom(wav_path)
        return {"status": "success", "temporary_chords": filtered_chords}

    except Exception as e:
        logging.error("Processing failed", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        for path in [raw_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)
