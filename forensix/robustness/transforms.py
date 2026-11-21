"""Realistic degradations used for robustness testing and the Forensic Lab.

Transforms operate on in-memory media (RGB uint8 image, float waveform, list of RGB frames) and are applied only
to TEST data, never to training data.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from ..preprocess.video import ffmpeg_h264_roundtrip


# ------------------------------------------------------------------ images
def jpeg_image(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def resize_image(img: np.ndarray, factor: float) -> np.ndarray:
    """Down-scale by ``factor`` then back up to the original size (what re-sharing does to a picture)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * factor)), max(8, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur_image(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), float(sigma))


def noise_image(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio
def bgnoise_audio(y: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(y**2)) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (y + rng.normal(0, np.sqrt(p_noise), y.shape)).astype(np.float32)


def mp3_audio(y: np.ndarray, sr: int, kbps: int) -> np.ndarray:
    """MP3 round trip via ffmpeg; falls back to a low-pass + re-quantisation approximation without it."""
    if shutil.which("ffmpeg") is None:
        return _lowpass_fallback(y, sr, kbps)
    import soundfile as sf

    d = tempfile.mkdtemp(prefix="fx_aud_")
    try:
        wav, mp3, back = (os.path.join(d, n) for n in ("a.wav", "a.mp3", "b.wav"))
        sf.write(wav, y, sr)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", wav, "-b:a", f"{int(kbps)}k", mp3], check=True, timeout=60
        )
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", mp3, "-ar", str(sr), "-ac", "1", back], check=True, timeout=60
        )
        out, _ = sf.read(back, dtype="float32")
        out = out[:, 0] if out.ndim > 1 else out
        return _align(out, len(y))
    except Exception:
        return _lowpass_fallback(y, sr, kbps)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _align(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def _lowpass_fallback(y: np.ndarray, sr: int, kbps: int) -> np.ndarray:
    spec = np.fft.rfft(y)
    cutoff = int(len(spec) * min(1.0, max(0.25, kbps / 128)))
    spec[cutoff:] = 0
    return np.fft.irfft(spec, len(y)).astype(np.float32)


def resample_audio(y: np.ndarray, sr: int, low_sr: int) -> np.ndarray:
    import librosa

    low = librosa.resample(y, orig_sr=sr, target_sr=int(low_sr))
    return _align(librosa.resample(low, orig_sr=int(low_sr), target_sr=sr), len(y)).astype(np.float32)


# ------------------------------------------------------------------ video
def h264_video(frames: list[np.ndarray], fps: float, crf: int) -> list[np.ndarray]:
    return ffmpeg_h264_roundtrip(frames, fps, crf)


def downscale_video(frames: list[np.ndarray], factor: float) -> list[np.ndarray]:
    return [resize_image(f, factor) for f in frames]


def fps_reduce_video(frames: list[np.ndarray], divisor: int) -> list[np.ndarray]:
    return frames[:: max(1, int(divisor))]


# ------------------------------------------------------------------ registry of named transforms
def image_transforms(cfg) -> dict[str, callable]:
    r = cfg["robustness"]["image"]
    t = {}
    for q in r["jpeg"]:
        t[f"JPEG q={q}"] = lambda x, q=q: jpeg_image(x, q)
    for f in r["resize"]:
        t[f"Resize x{f}"] = lambda x, f=f: resize_image(x, f)
    for s in r["blur"]:
        t[f"Blur sigma={s}"] = lambda x, s=s: blur_image(x, s)
    for s in r["noise"]:
        t[f"Noise sigma={s}"] = lambda x, s=s: noise_image(x, s)
    return t


def audio_transforms(cfg) -> dict[str, callable]:
    r = cfg["robustness"]["audio"]
    sr = cfg["audio"]["sr"]
    t = {}
    for s in r["bgnoise"]:
        t[f"Background noise {s} dB SNR"] = lambda y, s=s: bgnoise_audio(y, s)
    for k in r["mp3"]:
        t[f"MP3 {k} kbps"] = lambda y, k=k: mp3_audio(y, sr, k)
    for s in r["resample"]:
        t[f"Resample {s} Hz"] = lambda y, s=s: resample_audio(y, sr, s)
    return t


def video_transforms(cfg) -> dict[str, callable]:
    """Each takes ``(frames, fps)`` and returns the degraded frame list."""
    r = cfg["robustness"]["video"]
    t = {}
    for c in r["h264"]:
        t[f"H.264 CRF {c}"] = lambda fr, fps, c=c: h264_video(fr, fps, c)
    for f in r["downscale"]:
        t[f"Downscale x{f}"] = lambda fr, fps, f=f: downscale_video(fr, f)
    for d in r["fps_div"]:
        t[f"Frame rate /{d}"] = lambda fr, fps, d=d: fps_reduce_video(fr, d)
    return t


def transform_family(name: str) -> str:
    n = name.lower()
    for key, fam in (
        ("jpeg", "JPEG compression"),
        ("resize", "Resizing"),
        ("blur", "Blur"),
        ("noise sigma", "Noise"),
        ("background", "Background noise"),
        ("mp3", "Compression"),
        ("resample", "Resampling"),
        ("h.264", "Compression"),
        ("downscale", "Resolution reduction"),
        ("frame rate", "Frame-rate reduction"),
    ):
        if key in n:
            return fam
    return "Other"
