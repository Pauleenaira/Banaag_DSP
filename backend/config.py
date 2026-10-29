"""config.py - Central configuration and tunable settings for BANAAG.
Sourced from documentation.ipynb and optimized for Cebuano speech enhancement.
"""
import os

# Pipeline Sample Rate (Standard for Whisper and speech DSP)
SR = 16000

# Stage 3: Band-pass filter (Butterworth 4th-order, wideband speech 100-7000 Hz)
BPF = dict(lowcut=100, highcut=7000, order=4)

# Stage 4: Spectral subtraction (Trial B from notebook)
# alpha = over-subtraction factor, beta = spectral floor to prevent musical noise
SS = dict(alpha=2.0, beta=0.05)

# Stage 5: Decision-Directed Wiener filter (Trial A from notebook)
# smoothing = decision-directed smoothing factor, gain_floor = minimum gain floor
WF = dict(smoothing=0.60, gain_floor=0.10)

# STFT parameters shared across spectral subtraction, Wiener filter, and SNR estimation
NPERSEG = 512
NOVERLAP = 384
NOISE_PCT = 20  # Quietest 20% of frames used for noise estimation

# Whisper ASR Models (Cebuano fine-tuned models)
# Automatically selects Large v3 on CUDA GPU, Small on CPU to preserve system responsiveness
LARGE_MODEL_ID = "sapinsapin/whisper-large-v3-pld-ceb-norm"
LARGE_PROCESSOR_ID = "openai/whisper-large-v3"

SMALL_MODEL_ID = "sapinsapin/whisper-small-pld-ceb"
SMALL_PROCESSOR_ID = "openai/whisper-small"

CHUNK_SECONDS = 25  # Audio transcribed in 25-second windows
MAX_NEW_TOKENS = 225

# Directory Paths (Relative to repository root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = os.path.join(BASE_DIR, "work")
REFS_DIR = os.path.join(BASE_DIR, "references")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

