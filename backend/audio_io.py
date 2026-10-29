"""audio_io.py - Stages 1 and 2: Audio loading and standardization (16 kHz mono)."""
import io
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from config import SR


def load(source):
    """
    Read an audio file path or bytes.
    Returns (samples shaped [n, channels], original sample rate).
    Soundfile handles WAV, FLAC, OGG; falls back to librosa (+ ffmpeg) for M4A and others.
    """
    try:
        if isinstance(source, (bytes, bytearray)):
            bio = io.BytesIO(source)
            x, sr = sf.read(bio, dtype="float64", always_2d=True)
        else:
            x, sr = sf.read(source, dtype="float64", always_2d=True)
    except Exception:
        import librosa
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        x, sr = librosa.load(source, sr=None, mono=False)
        x = np.atleast_2d(x).T  # convert [channels, n] to [n, channels]
    return x, sr


def to_16k_mono(x, sr):
    """
    Average multi-channel audio to mono, then resample to 16 kHz using polyphase filtering.
    """
    mono = x.mean(axis=1) if x.ndim > 1 else x
    if sr != SR:
        g = gcd(sr, SR)
        mono = resample_poly(mono, SR // g, sr // g)
    return mono

