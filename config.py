"""config.py - every tunable setting in one place.
The numbers come from documentation.ipynb. Change them here, nowhere else."""

SR = 16000                      # sample rate the whole pipeline (and Whisper) works at

# Band-pass filter (notebook: Trial A, the root for spectral subtraction)
BPF = dict(lowcut=100, highcut=7000, order=4)

# Spectral subtraction (notebook: Trial B, the root for the Wiener filter)
#   alpha = how much of the noise estimate is subtracted, beta = spectral floor
SS = dict(alpha=2.0, beta=0.05)

# Wiener filter (notebook: Trial A, the one used in the WER evaluation cell)
#   smoothing = decision-directed smoothing, gain_floor = minimum gain
WF = dict(smoothing=0.60, gain_floor=0.10)

# STFT settings shared by spectral subtraction, Wiener and the SNR estimate
NPERSEG, NOVERLAP = 512, 384
NOISE_PCT = 20                  # the quietest 20% of frames are treated as "noise only"

# Whisper (Cebuano fine-tuned Whisper Small from Hugging Face)
MODEL_ID = "sapinsapin/whisper-small-pld-ceb"
CHUNK_SECONDS = 25              # audio is transcribed in 25 s windows
MAX_NEW_TOKENS = 225

WORK_DIR = "work"               # uploaded + cleaned files are stored here
REFS_DIR = "references"         # reference transcripts for automatic WER
