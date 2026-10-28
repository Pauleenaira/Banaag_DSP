"""audio_io.py - stages 1 and 2: load the file and standardize it (mono, 16 kHz)."""
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from config import SR


def load(path):
    """Read an audio file. Returns (samples shaped [n, channels], original sample rate).
    soundfile handles WAV/FLAC/MP3; anything else (e.g. m4a) falls back to librosa + ffmpeg."""
    try:
        x, sr = sf.read(path, dtype="float64", always_2d=True)
    except Exception:
        import librosa
        x, sr = librosa.load(path, sr=None, mono=False)
        x = np.atleast_2d(x).T          # librosa gives [channels, n]; we want [n, channels]
    return x, sr


def to_16k_mono(x, sr):
    """Average the channels to mono, then resample to 16 kHz (what Whisper expects)."""
    mono = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)                 # reduce the ratio, e.g. 48000/16000 -> 1/3
        mono = resample_poly(mono, SR // g, sr // g)
    return mono
