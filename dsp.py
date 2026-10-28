"""dsp.py - stages 3 to 5: the three noise-reduction filters, run in this order:
band-pass -> spectral subtraction -> Wiener."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft
from config import SR, BPF, SS, WF, NPERSEG, NOVERLAP, NOISE_PCT


def bandpass(audio):
    """Keep only 100-7000 Hz. sosfiltfilt runs the filter forward and backward,
    so the speech is not delayed or phase-shifted."""
    sos = butter(BPF["order"], [BPF["lowcut"], BPF["highcut"]],
                 btype="bandpass", fs=SR, output="sos")
    return sosfiltfilt(sos, audio)


def spectral_subtraction(audio):
    """Estimate the noise spectrum from the quietest frames and subtract it from every frame."""
    _, _, Z = stft(audio, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    mag, phase = np.abs(Z), np.angle(Z)

    # 1) find the quietest frames and average their spectrum = noise estimate
    energy = np.mean(mag ** 2, axis=0)
    quiet = energy <= np.percentile(energy, NOISE_PCT)
    noise = np.mean(mag[:, quiet], axis=1, keepdims=True)

    # 2) subtract alpha * noise, but never go below beta * original (avoids "musical noise")
    cleaned = np.maximum(mag - SS["alpha"] * noise, SS["beta"] * mag)

    # 3) rebuild the waveform using the ORIGINAL phase
    _, out = istft(cleaned * np.exp(1j * phase), fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    return out[:len(audio)]


def wiener_filter(audio):
    """Frame-by-frame Wiener gain using the decision-directed method:
    gain = prior_SNR / (1 + prior_SNR), where prior SNR blends the previous frame
    with the current one. gain_floor keeps speech from being muted completely."""
    _, _, Z = stft(audio, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    power = np.abs(Z) ** 2

    # noise power spectrum from the quietest frames (same idea as spectral subtraction)
    energy = np.mean(power, axis=0)
    quiet = energy <= np.percentile(energy, NOISE_PCT)
    noise_psd = np.maximum(np.mean(power[:, quiet], axis=1), 1e-12)

    out = np.zeros_like(Z)
    prev_gain = np.ones(len(noise_psd))
    prev_post = np.ones(len(noise_psd))
    a = WF["smoothing"]
    for t in range(Z.shape[1]):
        post = power[:, t] / noise_psd                                   # a-posteriori SNR
        prior = a * prev_gain ** 2 * prev_post + (1 - a) * np.maximum(post - 1, 0)
        gain = np.maximum(prior / (1 + prior), WF["gain_floor"])
        out[:, t] = gain * Z[:, t]
        prev_gain, prev_post = gain, post

    _, y = istft(out, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP)
    y = np.pad(y, (0, max(0, len(audio) - len(y))))[:len(audio)]         # match input length

    peak = np.max(np.abs(y))                                             # clipping guard
    return y * min(1.0, 0.99 / peak) if peak > 0 else y
