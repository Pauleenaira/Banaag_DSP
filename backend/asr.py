"""asr.py - Stage 6: Cebuano Speech-to-Text inference using fine-tuned Whisper."""
import os
import threading
import torch
import numpy as np
from config import (
    SR,
    CHUNK_SECONDS,
    MAX_NEW_TOKENS,
    LARGE_MODEL_ID,
    LARGE_PROCESSOR_ID,
    SMALL_MODEL_ID,
    SMALL_PROCESSOR_ID,
)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_processor = None
_current_model_id = None
_lock = threading.Lock()


def get_device_info():
    """Returns hardware acceleration details and active model information."""
    has_gpu = torch.cuda.is_available()
    return {
        "device": _device,
        "device_name": torch.cuda.get_device_name(0) if has_gpu else "Standard CPU",
        "has_gpu": has_gpu,
        "active_model": _current_model_id or (LARGE_MODEL_ID if has_gpu else SMALL_MODEL_ID),
        "is_loaded": _model is not None and _processor is not None,
    }


def get_model(force_small=False):
    """
    Loads Whisper once and caches it in memory.
    - If CUDA is available: uses Whisper Large v3 (FP16).
    - If CPU (laptops): uses Whisper Small (FP32) to prevent freezing and save RAM.
    """
    global _model, _processor, _current_model_id
    with _lock:
        if force_small or _device != "cuda":
            target_model = SMALL_MODEL_ID
            target_proc = SMALL_PROCESSOR_ID
            dtype = torch.float32
        else:
            target_model = LARGE_MODEL_ID
            target_proc = LARGE_PROCESSOR_ID
            dtype = torch.float16

        if _model is None or _current_model_id != target_model:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            print(f"[ASR] Device: {_device.upper()} ({get_device_info()['device_name']})")
            print(f"[ASR] Loading processor '{target_proc}'...")
            _processor = WhisperProcessor.from_pretrained(target_proc)

            print(f"[ASR] Loading Cebuano Whisper model '{target_model}'...")
            if _device == "cpu":
                torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

            _model = WhisperForConditionalGeneration.from_pretrained(
                target_model,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(_device)
            _model.eval()
            _current_model_id = target_model
            print(f"[ASR] Model loaded successfully on {_device.upper()}!")

    return _processor, _model, _device, dtype


def transcribe(audio):
    """
    Transcribe 16 kHz mono audio in 25-second chunks.
    Returns: [{"start": float, "end": float, "text": str}, ...]
    """
    processor, model, device, dtype = get_model()
    chunk_samples = CHUNK_SECONDS * SR
    segments = []

    for start in range(0, len(audio), chunk_samples):
        piece = audio[start : start + chunk_samples].astype("float32")
        if len(piece) < SR // 10:  # skip slices shorter than 0.1 s
            continue

        inputs = processor(piece, sampling_rate=SR, return_tensors="pt")
        input_features = inputs.input_features.to(device, dtype=dtype)

        with torch.inference_mode():
            ids = model.generate(
                input_features,
                task="transcribe",
                max_new_tokens=MAX_NEW_TOKENS,
            )

        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        segments.append({
            "start": round(start / SR, 2),
            "end": round(min(len(audio), start + chunk_samples) / SR, 2),
            "text": text,
        })

    return segments
