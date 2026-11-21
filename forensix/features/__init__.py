"""Feature extraction entry points shared by training, robustness testing and the web app."""

from __future__ import annotations

import numpy as np

from ..preprocess import audio as pa
from ..preprocess import image as pi
from ..preprocess import video as pv
from ..utils import sanitize_features
from .audio import extract_audio_features
from .image import extract_image_features
from .video import extract_video_features


# ---- from in-memory objects (used by robustness transforms) --------------------------------
def features_from_image_array(rgb_uint8: np.ndarray, cfg) -> dict[str, float]:
    img = pi.preprocess_image(rgb_uint8, cfg.get("image", "size"))
    return sanitize_features(extract_image_features(img))


def features_from_waveform(y: np.ndarray, cfg) -> dict[str, float] | None:
    a = cfg["audio"]
    ok, _ = pa.validate_audio(y, a["sr"], a["min_seconds"], a["silence_db"])
    if not ok:
        return None
    y = pa.normalise_amplitude(y)
    segs = pa.segment_audio(y, a["sr"], a["segment_seconds"], a["max_segments"])
    rows = [extract_audio_features(s, a["sr"]) for s in segs]
    keys = rows[0].keys()
    return sanitize_features({k: float(np.mean([r[k] for r in rows])) for k in keys})


def features_from_frames(frames: list[np.ndarray], cfg) -> dict[str, float] | None:
    v = cfg["video"]
    arr = pv.sample_frames(frames, v["frame_interval"], v["max_frames"], v["size"])
    if arr.shape[0] < v["min_frames"]:
        return None
    return sanitize_features(extract_video_features(arr))


# ---- from files -----------------------------------------------------------------------------
def load_media(path, modality: str, cfg):
    """Load a file into the raw in-memory object for its modality (None if unreadable)."""
    if modality == "image":
        return pi.load_image(path)
    if modality == "audio":
        return pa.load_audio(path, cfg["audio"]["sr"])
    if modality == "video":
        res = pv.read_raw_frames(path, cfg["video"]["max_raw_frames"], cfg["video"]["max_side"])
        return res  # (frames, fps) or None
    raise ValueError(f"unknown modality {modality!r}")


def features_from_media(media, modality: str, cfg) -> dict[str, float] | None:
    if media is None:
        return None
    if modality == "image":
        return features_from_image_array(media, cfg)
    if modality == "audio":
        return features_from_waveform(media, cfg)
    if modality == "video":
        return features_from_frames(media[0], cfg)
    raise ValueError(modality)


def extract_from_path(path, modality: str, cfg) -> dict[str, float] | None:
    """Full path -> feature dict, or None if the file is unusable."""
    try:
        return features_from_media(load_media(path, modality, cfg), modality, cfg)
    except Exception:
        return None
