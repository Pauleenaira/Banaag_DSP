"""main.py - Entry point and FastAPI server for BANAAG.
Orchestrates the 7-stage DSP enhancement and Whisper AI transcription pipeline.
Run: python main.py
"""
import os
import sys
import uuid
import time
import threading
from contextlib import asynccontextmanager
from typing import Optional

# Ensure backend directory is in the Python module search path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import SR, WORK_DIR, REFS_DIR, FRONTEND_DIR
from audio_io import load, to_16k_mono
from dsp import bandpass, spectral_subtraction, wiener_filter
import asr
from metrics import wer_percent, estimate_snr, find_reference

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(REFS_DIR, exist_ok=True)

# In-memory dictionary tracking job status and progress
JOBS: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm Whisper model in a background thread at startup
    threading.Thread(target=asr.get_model, daemon=True).start()
    yield


app = FastAPI(
    title="BANAAG DSP Audio Enhancement & Whisper AI API",
    description="Speech Enhancement and Whisper AI Evaluation for Cebuano Audio",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def step(job: dict, fn, *args):
    """
    Executes one pipeline stage:
    Checks for cancellation, measures elapsed duration, and advances stage count.
    """
    if job.get("cancel"):
        raise RuntimeError("Cancelled")
    started = time.time()
    result = fn(*args)
    job["times"].append(round(time.time() - started, 2))
    job["stage"] += 1
    return result


def run_job(job_id: str, path: str, reference: Optional[str], reference_name: Optional[str]):
    """
    Runs the 7-stage processing pipeline in a background worker thread:
    1. Load audio
    2. Convert to 16 kHz mono
    3. Band-pass filter
    4. Spectral subtraction
    5. Wiener filter
    6. Whisper transcription (enhanced + baseline)
    7. Calculate metrics (SNR and WER)
    """
    job = JOBS[job_id]
    try:
        raw, sr = step(job, load, path)                         # Stage 1: Load
        mono = step(job, to_16k_mono, raw, sr)                  # Stage 2: 16k mono
        banded = step(job, bandpass, mono)                      # Stage 3: Butterworth BPF
        reduced = step(job, spectral_subtraction, banded)       # Stage 4: Spectral subtraction
        cleaned = step(job, wiener_filter, reduced)             # Stage 5: Wiener filter

        # Save processed WAV for client playback and download
        out_path = os.path.join(WORK_DIR, f"{job_id}.wav")
        sf.write(out_path, cleaned, SR, subtype="PCM_16")

        def asr_stage():                                        # Stage 6: Whisper ASR
            segments = asr.transcribe(cleaned)
            baseline = asr.transcribe(mono) if reference else None
            return segments, baseline

        segments, baseline = step(job, asr_stage)

        def score_stage():                                      # Stage 7: SNR & WER
            snr_cleaned = estimate_snr(cleaned)
            snr_mono = estimate_snr(mono)
            r = {
                "snr_db": round(snr_cleaned, 1),
                "snr_note": f"Estimated · was {snr_mono:.1f} dB before cleaning",
                "wer": None,
                "wer_baseline": None,
                "ref_name": reference_name,
                "segments": segments,
                "audio_url": f"/api/jobs/{job_id}/audio",
            }
            if reference:
                r["wer"] = wer_percent(reference, segments)
                if baseline:
                    r["wer_baseline"] = wer_percent(reference, baseline)
            return r

        job["result"] = step(job, score_stage)
        job["done"] = True

    except Exception as e:
        job["error"] = str(e)


# Serve Frontend Web UI
@app.get("/", response_class=FileResponse)
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="frontend/index.html not found.")


@app.get("/api/asr/status")
def get_status():
    """Returns GPU/CPU hardware info and active Whisper model status."""
    return asr.get_device_info()


@app.post("/api/jobs", status_code=status.HTTP_200_OK)
async def create_job(
    file: UploadFile = File(..., description="Audio file to enhance and transcribe"),
    reference: Optional[UploadFile] = File(None, description="Optional ground truth transcript (.txt)"),
):
    """
    Receives upload, matches or parses reference transcript, and begins 7-stage processing.
    """
    job_id = uuid.uuid4().hex[:10]
    path = os.path.join(WORK_DIR, f"{job_id}_in")

    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    ref_text = None
    ref_name = None

    if reference:
        ref_bytes = await reference.read()
        ref_text = ref_bytes.decode("utf-8", errors="replace").strip()
        ref_name = reference.filename
    else:
        ref_name, ref_text = find_reference(file.filename)

    JOBS[job_id] = {
        "stage": 0,
        "times": [],
        "done": False,
        "cancel": False,
        "error": None,
        "result": None,
    }

    threading.Thread(
        target=run_job,
        args=(job_id, path, ref_text, ref_name),
        daemon=True,
    ).start()

    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    """
    Polled by the frontend processing view (every 600 ms) to update the 7-stage checklist.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str):
    """Streams the enhanced 16 kHz WAV audio back to the client."""
    audio_path = os.path.join(WORK_DIR, f"{job_id}.wav")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found or processing not complete")
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"banaag_{job_id}_cleaned.wav",
    )


@app.delete("/api/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def cancel_job(job_id: str):
    """Cancels a currently active job."""
    if job_id in JOBS:
        JOBS[job_id]["cancel"] = True
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# Mount static assets if present in frontend/
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)

