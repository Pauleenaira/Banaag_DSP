"""metrics.py - Stage 7: Word Error Rate (WER), estimated SNR, and reference file matching."""
import os
import re
import numpy as np
from scipy.signal import stft
from config import SR, NPERSEG, NOVERLAP, NOISE_PCT, REFS_DIR


def normalize(text):
    """Lowercase and strip punctuation to ensure fair WER calculation."""
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def wer_percent(reference, hypothesis):
    """
    Computes Word Error Rate (%) = (Substitutions + Deletions + Insertions) / Reference Words.
    Hypothesis can be a list of segment dicts or a string.
    """
    from jiwer import wer

    if isinstance(hypothesis, list):
        hyp_text = " ".join(s.get("text", "") for s in hypothesis)
    else:
        hyp_text = str(hypothesis)

    ref_norm = normalize(reference)
    hyp_norm = normalize(hyp_text)

    if not ref_norm:
        return 0.0

    return round(float(wer(ref_norm, hyp_norm) * 100), 2)


def estimate_snr(audio):
    """
    Estimates SNR in dB based on the STFT quietest 20% frames vs. overall speech power.
    """
    _, _, Z = stft(audio, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(Z) ** 2
    energy = power.mean(axis=0)

    noise_frames = energy <= np.percentile(energy, NOISE_PCT)
    noise_power = power[:, noise_frames].mean()
    speech_power = max(power.mean() - noise_power, 1e-12)

    return float(10 * np.log10(speech_power / max(noise_power, 1e-12)))


def find_reference(upload_name):
    """
    Scans REFS_DIR for a reference text file matching the uploaded audio's filename.
    Matches e.g. CEB_006_reference.txt for CEB_006_train_15dB.wav.
    Returns (filename, text) or (None, None).
    """
    if not os.path.exists(REFS_DIR) or not upload_name:
        return None, None

    stem = os.path.splitext(upload_name)[0].lower()
    for f in sorted(os.listdir(REFS_DIR)):
        base, ext = os.path.splitext(f)
        if ext.lower() != ".txt":
            continue
        key = re.sub(r"_?reference$", "", base.lower())
        if key and stem.startswith(key):
            try:
                with open(os.path.join(REFS_DIR, f), encoding="utf-8") as fh:
                    return f, fh.read().strip()
            except Exception:
                pass
    return None, None

