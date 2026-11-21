"""Image preprocessing: corruption check, common resolution, RGB colour space, [0, 1] normalisation."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path) -> np.ndarray | None:
    """Read an image as an RGB uint8 array, or None if the file is corrupt / unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3 or min(bgr.shape[:2]) < 8:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def resize_to(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interp)


def preprocess_image(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """RGB uint8 -> float32 in [0, 1] at a common ``size`` x ``size`` resolution."""
    out = resize_to(rgb, size).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


def dhash(rgb: np.ndarray, hash_size: int = 8) -> str:
    """Difference hash used to flag exact / near-exact duplicates."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)
