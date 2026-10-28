"""metrics.py - stage 7: WER, estimated SNR and automatic reference lookup."""
import os, re
import numpy as np
from scipy.signal import stft
from config import SR, NPERSEG, NOVERLAP, NOISE_PCT, REFS_DIR


def normalize(text):
    """Lowercase and strip punctuation so 'Maayo,' and 'maayo' count as the same word."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def wer_percent(reference, segments):
    """Word Error Rate (%) = (substitutions + deletions + insertions) / reference words."""
    from jiwer import wer
    hypothesis = " ".join(s["text"] for s in segments)
    return float(wer(normalize(reference), normalize(hypothesis)) * 100)


def estimate_snr(audio):
    """Estimated SNR in dB. Your notebook compares against clean speech, which an uploaded
    file does not have, so noise power is taken from the quietest frames instead.
    It is an estimate, and it tends to flatter audio that has been noise-reduced."""
    _, _, Z = stft(audio, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(Z) ** 2
    energy = power.mean(axis=0)
    noise = power[:, energy <= np.percentile(energy, NOISE_PCT)].mean()
    speech = max(power.mean() - noise, 1e-12)
    return float(10 * np.log10(speech / max(noise, 1e-12)))


def find_reference(upload_name):
    """Look in references/ for a transcript that matches the uploaded file name.
    Save it as <key>_reference.txt; any upload starting with <key> uses it, e.g.
    CEB_006_reference.txt matches CEB_006_train_15dB.wav.
    Returns (file name, text) or (None, None)."""
    stem = os.path.splitext(upload_name)[0].lower()
    for f in sorted(os.listdir(REFS_DIR)):
        base, ext = os.path.splitext(f)
        if ext.lower() != ".txt":
            continue
        key = re.sub(r"_?reference$", "", base.lower())
        if key and stem.startswith(key):
            with open(os.path.join(REFS_DIR, f), encoding="utf-8") as fh:
                return f, fh.read().strip()
    return None, None
