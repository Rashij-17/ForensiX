"""Audio preprocessing: validation, resampling, amplitude normalisation, silence checks, segmentation."""

from __future__ import annotations

import numpy as np


def load_audio(path, sr: int = 16000) -> np.ndarray | None:
    """Load mono audio resampled to ``sr``. Returns None when the file cannot be decoded."""
    try:
        import librosa

        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception:
        return None


def rms_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y)) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def validate_audio(y: np.ndarray | None, sr: int, min_seconds: float, silence_db: float) -> tuple[bool, str]:
    """Return (ok, reason). Rejects undecodable, too-short, non-finite and silent recordings."""
    if y is None or y.size == 0:
        return False, "undecodable"
    if not np.all(np.isfinite(y)):
        return False, "non_finite"
    if y.size / sr < min_seconds:
        return False, "too_short"
    if rms_db(y) < silence_db:
        return False, "silent"
    return True, "ok"


def normalise_amplitude(y: np.ndarray) -> np.ndarray:
    """Remove DC offset and peak-normalise to 0.95."""
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    return (y / peak * 0.95).astype(np.float32) if peak > 1e-9 else y.astype(np.float32)


def segment_audio(y: np.ndarray, sr: int, segment_seconds: float, max_segments: int) -> list[np.ndarray]:
    """Cut into non-overlapping segments; a short tail (<50 % of a segment) is dropped."""
    seg_len = int(segment_seconds * sr)
    if y.size <= seg_len:
        # Short clips are analysed as-is: padding would invent silence that dominates the statistics.
        return [y]
    segs = []
    for start in range(0, y.size, seg_len):
        chunk = y[start : start + seg_len]
        if chunk.size >= 0.5 * seg_len:
            segs.append(chunk if chunk.size == seg_len else np.pad(chunk, (0, seg_len - chunk.size)))
        if len(segs) >= max_segments:
            break
    return segs or [y]
