"""asr.py - stage 6: speech-to-text with the Cebuano Whisper model."""
import threading
from config import SR, MODEL_ID, CHUNK_SECONDS, MAX_NEW_TOKENS

_model = None
_lock = threading.Lock()                # stops two threads from loading the model twice


def get_model():
    """Load Whisper once and reuse it. The first call downloads about 1 GB, so app.py
    calls this in the background at startup. Uses the GPU when one is available."""
    global _model
    with _lock:
        if _model is None:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = WhisperProcessor.from_pretrained(MODEL_ID)
            model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()
            _model = (processor, model, device, torch)
    return _model


def transcribe(audio):
    """Transcribe 16 kHz mono audio in 25 s chunks.
    Returns [{"start": sec, "end": sec, "text": str}, ...] (one entry per chunk)."""
    processor, model, device, torch = get_model()
    chunk = CHUNK_SECONDS * SR
    segments = []
    for start in range(0, len(audio), chunk):
        piece = audio[start:start + chunk].astype("float32")
        if len(piece) < SR // 10:       # skip a tail shorter than 0.1 s
            continue
        feats = processor(piece, sampling_rate=SR, return_tensors="pt").input_features.to(device)
        with torch.inference_mode():
            ids = model.generate(feats, task="transcribe", max_new_tokens=MAX_NEW_TOKENS)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        segments.append({"start": start / SR,
                         "end": min(len(audio), start + chunk) / SR,
                         "text": text})
    return segments
