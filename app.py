"""app.py - the web server. It only wires things together:
the filters are in dsp.py, Whisper in asr.py, scoring in metrics.py.

Run:  python app.py      then open  http://localhost:5000

API used by index.html
  POST   /api/jobs             upload "file" (+ optional "reference") -> {"id": ...}
  GET    /api/jobs/<id>        progress: {stage, times, done, error, result}
  GET    /api/jobs/<id>/audio  the cleaned WAV
  DELETE /api/jobs/<id>        cancel
"""
import os, uuid, threading, time
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory

from config import SR, WORK_DIR, REFS_DIR
from audio_io import load, to_16k_mono
from dsp import bandpass, spectral_subtraction, wiener_filter
from asr import get_model, transcribe
from metrics import wer_percent, estimate_snr, find_reference

app = Flask(__name__, static_folder=".", static_url_path="")
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(REFS_DIR, exist_ok=True)
JOBS = {}                                # job id -> progress dict (kept in memory)


def step(job, fn, *args):
    """Run one pipeline stage: stop if cancelled, time it, and advance the progress counter.
    The UI shows 7 stages, so keep exactly 7 step() calls per job."""
    if job["cancel"]:
        raise RuntimeError("Cancelled")
    started = time.time()
    result = fn(*args)
    job["times"].append(time.time() - started)
    job["stage"] += 1
    return result


def run_job(job_id, path, reference, reference_name):
    """The whole pipeline for one upload. Runs in a background thread."""
    job = JOBS[job_id]
    try:
        raw, sr = step(job, load, path)                          # stage 1: load
        mono = step(job, to_16k_mono, raw, sr)                   # stage 2: mono + 16 kHz
        banded = step(job, bandpass, mono)                       # stage 3
        reduced = step(job, spectral_subtraction, banded)        # stage 4
        cleaned = step(job, wiener_filter, reduced)              # stage 5

        out_path = os.path.join(WORK_DIR, job_id + ".wav")
        sf.write(out_path, cleaned, SR, subtype="PCM_16")        # file the browser plays/downloads

        def asr_stage():                                         # stage 6: Whisper
            segments = transcribe(cleaned)
            baseline = transcribe(mono) if reference else None   # original audio, for "was X% before"
            return segments, baseline
        segments, baseline = step(job, asr_stage)

        def score_stage():                                       # stage 7: metrics
            r = {"snr_db": estimate_snr(cleaned),
                 "snr_note": f"Estimated · was {estimate_snr(mono):.1f} dB before cleaning",
                 "wer": None, "wer_baseline": None, "ref_name": reference_name,
                 "segments": segments, "audio_url": f"/api/jobs/{job_id}/audio"}
            if reference:
                r["wer"] = wer_percent(reference, segments)
                r["wer_baseline"] = wer_percent(reference, baseline)
            return r
        job["result"] = step(job, score_stage)
        job["done"] = True
    except Exception as e:
        job["error"] = str(e)


@app.post("/api/jobs")
def create_job():
    """Receive the upload, pick a reference transcript, and start processing."""
    job_id = uuid.uuid4().hex[:10]
    path = os.path.join(WORK_DIR, job_id + "_in")
    request.files["file"].save(path)
    if "reference" in request.files:                             # manual upload wins
        rf = request.files["reference"]
        reference, reference_name = rf.read().decode("utf-8").strip(), rf.filename
    else:                                                        # else look in references/
        reference_name, reference = find_reference(request.files["file"].filename)
    JOBS[job_id] = {"stage": 0, "times": [], "done": False, "cancel": False}
    threading.Thread(target=run_job, args=(job_id, path, reference, reference_name),
                     daemon=True).start()
    return jsonify(id=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    """Polled by the processing screen every 0.6 s."""
    job = JOBS.get(job_id)
    return (jsonify(job), 200) if job else ("", 404)


@app.get("/api/jobs/<job_id>/audio")
def job_audio(job_id):
    return send_from_directory(WORK_DIR, job_id + ".wav")


@app.delete("/api/jobs/<job_id>")
def cancel_job(job_id):
    if job_id in JOBS:
        JOBS[job_id]["cancel"] = True
    return "", 204


@app.get("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    # Load Whisper in the background so the first upload does not wait for the download.
    threading.Thread(target=get_model, daemon=True).start()
    app.run(host="127.0.0.1", port=5000)
