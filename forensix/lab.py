"""Forensic Lab (stress-test one sample) and Compare (two versions of the same media)."""

from __future__ import annotations


from .features import features_from_frames, features_from_image_array, features_from_waveform
from .robustness import transforms as T


def lab_options(modality: str) -> dict[str, dict]:
    """Adjustable degradations offered per modality: name -> {label, min, max, default, step}."""
    if modality == "image":
        return {
            "jpeg": {"label": "JPEG quality", "min": 10, "max": 95, "default": 50, "step": 5},
            "resize": {"label": "Resize factor", "min": 0.2, "max": 0.95, "default": 0.5, "step": 0.05},
            "blur": {"label": "Gaussian blur sigma", "min": 0.3, "max": 4.0, "default": 1.5, "step": 0.1},
            "noise": {"label": "Gaussian noise sigma", "min": 1, "max": 25, "default": 8, "step": 1},
        }
    if modality == "audio":
        return {
            "bgnoise": {
                "label": "Background noise SNR (dB, lower = noisier)",
                "min": 0,
                "max": 40,
                "default": 20,
                "step": 1,
            },
            "mp3": {"label": "MP3 bitrate (kbps)", "min": 16, "max": 128, "default": 48, "step": 8},
            "resample": {"label": "Resample to (Hz)", "min": 4000, "max": 12000, "default": 8000, "step": 500},
        }
    return {
        "h264": {"label": "H.264 CRF (higher = more compression)", "min": 20, "max": 45, "default": 33, "step": 1},
        "downscale": {"label": "Resolution factor", "min": 0.2, "max": 0.95, "default": 0.5, "step": 0.05},
        "fps_div": {"label": "Keep every n-th frame", "min": 2, "max": 4, "default": 2, "step": 1},
    }


def apply_transform(modality: str, media, kind: str, strength: float, cfg):
    """Return the degraded media (same structure as the input)."""
    if modality == "image":
        fn = {
            "jpeg": lambda x: T.jpeg_image(x, int(strength)),
            "resize": lambda x: T.resize_image(x, float(strength)),
            "blur": lambda x: T.blur_image(x, float(strength)),
            "noise": lambda x: T.noise_image(x, float(strength)),
        }[kind]
        return fn(media)
    if modality == "audio":
        sr = cfg["audio"]["sr"]
        fn = {
            "bgnoise": lambda y: T.bgnoise_audio(y, float(strength)),
            "mp3": lambda y: T.mp3_audio(y, sr, int(strength)),
            "resample": lambda y: T.resample_audio(y, sr, int(strength)),
        }[kind]
        return fn(media)
    frames, fps = media
    fn = {
        "h264": lambda: T.h264_video(frames, fps, int(strength)),
        "downscale": lambda: T.downscale_video(frames, float(strength)),
        "fps_div": lambda: T.fps_reduce_video(frames, int(strength)),
    }[kind]
    return fn(), fps


def media_features(modality: str, media, cfg) -> dict | None:
    if modality == "image":
        return features_from_image_array(media, cfg)
    if modality == "audio":
        return features_from_waveform(media, cfg)
    return features_from_frames(media[0], cfg)


def stress_curve(engine, modality: str, media, kind: str, strengths: list[float]) -> list[dict]:
    """Model confidence at several strengths of one degradation (plus the untouched original)."""
    cfg = engine.cfg
    base = media_features(modality, media, cfg)
    p0 = engine.score_features(base, modality)
    rows = [{"strength": None, "label": "original", "p_manipulated": p0, "delta": 0.0, "analysable": True}]
    for s in strengths:
        try:
            feats = media_features(modality, apply_transform(modality, media, kind, s, cfg), cfg)
        except Exception:
            feats = None
        if feats is None:
            rows.append({"strength": s, "label": str(s), "p_manipulated": None, "delta": None, "analysable": False})
        else:
            p = engine.score_features(feats, modality)
            rows.append({"strength": s, "label": str(s), "p_manipulated": p, "delta": p - p0, "analysable": True})
    return rows


def compare_scores(engine, modality: str, media_a, media_b) -> dict:
    cfg = engine.cfg
    fa, fb = media_features(modality, media_a, cfg), media_features(modality, media_b, cfg)
    if fa is None or fb is None:
        raise ValueError("one of the two files is too short or too degraded to analyse")
    pa, pb = engine.score_features(fa, modality), engine.score_features(fb, modality)
    shifts = sorted(((k, fb[k] - fa[k], (fb[k] - fa[k]) / (abs(fa[k]) + 1e-9)) for k in fa), key=lambda t: -abs(t[2]))[
        :8
    ]
    return {
        "p_a": pa,
        "p_b": pb,
        "delta": pb - pa,
        "largest_feature_shifts": [
            {"feature": k, "abs_change": float(d), "rel_change": float(r)} for k, d, r in shifts
        ],
    }
