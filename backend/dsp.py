"""dsp.py - Stages 3 to 5: The three cascaded noise-reduction DSP filters:
Butterworth Band-pass -> STFT Spectral Subtraction -> Decision-Directed Wiener Filter.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft
from config import SR, BPF, SS, WF, NPERSEG, NOVERLAP, NOISE_PCT


def bandpass(audio):
    """
    Stage 3: 4th-order Butterworth bandpass filter (100 Hz - 7000 Hz).
    Uses sosfiltfilt for zero-phase forward-backward filtering to avoid phase delay.
    """
    sos = butter(BPF["order"], [BPF["lowcut"], BPF["highcut"]],
                 btype="bandpass", fs=SR, output="sos")
    return sosfiltfilt(sos, audio)


def spectral_subtraction(audio):
    """
    Stage 4: Magnitude spectral subtraction in the STFT domain.
    Estimates the noise spectrum from the quietest 20% frames and subtracts alpha * noise,
    clamped to a spectral floor beta * original magnitude to avoid musical noise.
    """
    _, _, Z = stft(audio, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    mag, phase = np.abs(Z), np.angle(Z)

    # Estimate noise from the quietest frames
    energy = np.mean(mag ** 2, axis=0)
    quiet = energy <= np.percentile(energy, NOISE_PCT)
    noise = np.mean(mag[:, quiet], axis=1, keepdims=True)

    # Subtract noise with spectral floor
    cleaned = np.maximum(mag - SS["alpha"] * noise, SS["beta"] * mag)

    # Reconstruct waveform with original phase
    _, out = istft(cleaned * np.exp(1j * phase), fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    return out[:len(audio)]


def wiener_filter(audio):
    """
    Stage 5: Frame-by-frame adaptive Wiener filter using Ephraim-Malah decision-directed prior SNR:
    prior_SNR = a * prev_gain^2 * prev_post + (1 - a) * max(post - 1, 0)
    gain = max(prior / (1 + prior), gain_floor)
    """
    _, _, Z = stft(audio, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(Z) ** 2

    # Noise power spectral density from quietest frames
    energy = np.mean(power, axis=0)
    quiet = energy <= np.percentile(energy, NOISE_PCT)
    noise_psd = np.maximum(np.mean(power[:, quiet], axis=1), 1e-12)

    out = np.zeros_like(Z)
    prev_gain = np.ones(len(noise_psd))
    prev_post = np.ones(len(noise_psd))
    a = WF["smoothing"]
    floor = WF["gain_floor"]

    for t in range(Z.shape[1]):
        post = power[:, t] / noise_psd
        prior = a * (prev_gain ** 2) * prev_post + (1 - a) * np.maximum(post - 1, 0)
        gain = np.maximum(prior / (1 + prior), floor)
        out[:, t] = gain * Z[:, t]
        prev_gain, prev_post = gain, post

    _, y = istft(out, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    y = np.pad(y, (0, max(0, len(audio) - len(y))))[:len(audio)]

    # Peak clipping guard
    peak = np.max(np.abs(y))
    return y * min(1.0, 0.99 / peak) if peak > 0 else y
