"""Audio forensic features: MFCC, spectral shape, zero-crossing rate, energy and pitch statistics."""

from __future__ import annotations

import numpy as np

N_FFT = 512
HOP = 128


def _ms(prefix: str, x: np.ndarray, out: dict) -> None:
    out[f"{prefix}_mean"] = float(np.mean(x))
    out[f"{prefix}_std"] = float(np.std(x))


def extract_audio_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """``y`` is a mono, amplitude-normalised waveform. Returns an ordered feature dict."""
    import librosa

    f: dict[str, float] = {}
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) + 1e-10
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    for i in range(13):
        f[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        f[f"mfcc{i + 1}_std"] = float(mfcc[i].std())
    delta = librosa.feature.delta(mfcc)
    f["mfcc_delta_std_mean"] = float(delta.std(axis=1).mean())

    _ms("spec_centroid", librosa.feature.spectral_centroid(S=S, sr=sr)[0], f)
    _ms("spec_bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)[0], f)
    _ms("spec_rolloff", librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0], f)
    f["spec_flatness_mean"] = float(librosa.feature.spectral_flatness(S=S).mean())
    f["spec_contrast_mean"] = float(librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=N_FFT).mean())
    flux = np.sqrt((np.diff(S, axis=1) ** 2).sum(axis=0))
    f["spec_flux_std"] = float(flux.std() / (flux.mean() + 1e-9))

    _ms("zcr", librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0], f)

    rms = librosa.feature.rms(S=S, frame_length=N_FFT)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    _ms("rms", rms, f)
    f["rms_dyn_range_db"] = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    f["noise_floor_db"] = float(np.percentile(rms_db, 10))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    power = (S**2).sum(axis=1)
    tot = power.sum() + 1e-12
    f["hf_energy_ratio_4k"] = float(power[freqs >= 4000].sum() / tot)
    f["hf_energy_ratio_7k"] = float(power[freqs >= min(7000, 0.875 * sr / 2)].sum() / tot)

    try:
        f0 = librosa.yin(y, fmin=70, fmax=400, sr=sr, frame_length=1024)
        plausible = (f0 > 70 * 1.05) & (f0 < 400 * 0.95)
        f0v = f0[plausible]
        if f0v.size > 4:
            f["pitch_mean"] = float(f0v.mean())
            f["pitch_std"] = float(f0v.std())
            f["pitch_jitter"] = float(np.mean(np.abs(np.diff(f0v))) / (f0v.mean() + 1e-9))
        else:
            f["pitch_mean"] = f["pitch_std"] = f["pitch_jitter"] = 0.0
        f["pitch_plausible_frac"] = float(plausible.mean())
    except Exception:
        f["pitch_mean"] = f["pitch_std"] = f["pitch_jitter"] = f["pitch_plausible_frac"] = 0.0
    return f
