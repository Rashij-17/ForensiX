"""Video forensic features: frame-level statistics, frame-to-frame differences, motion consistency."""

from __future__ import annotations

import cv2
import numpy as np

from .image import extract_image_features, noise_residual, to_gray

FRAME_LEVEL_FEATURES = (
    "gray_var",
    "gray_entropy",
    "edge_density",
    "laplacian_var",
    "noise_std",
    "fft_hf_ratio",
    "dct_hf_ratio",
    "noise_autocorr_h1",
)


def _flow_stats(gray8: np.ndarray) -> dict[str, np.ndarray]:
    """Per consecutive pair: mean flow magnitude, mean flow vector, flow-compensated warp error."""
    mags, vecs, errs, spatial = [], [], [], []
    h, w = gray8[0].shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    for a, b in zip(gray8[:-1], gray8[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.hypot(flow[..., 0], flow[..., 1])
        mags.append(mag.mean())
        spatial.append(mag.std())
        vecs.append(flow.reshape(-1, 2).mean(axis=0))
        warped = cv2.remap(b, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        errs.append(np.abs(warped.astype(np.float32) - a.astype(np.float32)).mean() / 255.0)
    return {"mag": np.array(mags), "vec": np.array(vecs), "err": np.array(errs), "spatial": np.array(spatial)}


def extract_video_features(frames: np.ndarray) -> dict[str, float]:
    """``frames`` is float32 (T, H, W, 3) in [0, 1]. Aggregates everything into one video-level vector."""
    f: dict[str, float] = {}
    T = frames.shape[0]
    gray = np.stack([to_gray(fr) for fr in frames])

    # ---- frame-level visual statistics (mean / std over time) ---------
    per_frame = [extract_image_features(fr) for fr in frames]
    for name in FRAME_LEVEL_FEATURES:
        vals = np.array([pf[name] for pf in per_frame])
        f[f"frame_{name}_mean"] = float(vals.mean())
        f[f"frame_{name}_std"] = float(vals.std())
    bright = gray.mean(axis=(1, 2))
    f["brightness_std"] = float(bright.std())
    f["brightness_change_mean"] = float(np.abs(np.diff(bright)).mean()) if T > 1 else 0.0
    f["brightness_change_max"] = float(np.abs(np.diff(bright)).max()) if T > 1 else 0.0

    # ---- frame-to-frame differences ------------------------------------
    diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2)) if T > 1 else np.zeros(1)
    f["diff_mean"] = float(diffs.mean())
    f["diff_std"] = float(diffs.std())
    f["diff_max"] = float(diffs.max())
    f["diff_cv"] = float(diffs.std() / (diffs.mean() + 1e-9))
    f["diff_jerk"] = float(np.abs(np.diff(diffs)).mean()) if diffs.size > 1 else 0.0

    # ---- texture & noise change over time --------------------------------
    lap = np.array([pf["laplacian_var"] for pf in per_frame])
    f["texture_change"] = float(np.abs(np.diff(lap)).mean() / (lap.mean() + 1e-9)) if T > 1 else 0.0
    res = np.stack([noise_residual(g) for g in gray])
    corrs = []
    for a, b in zip(res[:-1], res[1:]):
        a0, b0 = a - a.mean(), b - b.mean()
        corrs.append(float((a0 * b0).sum() / (np.sqrt((a0**2).sum() * (b0**2).sum()) + 1e-12)))
    f["noise_temporal_corr"] = float(np.mean(corrs)) if corrs else 0.0

    # ---- motion consistency (optical flow) ---------------------------------
    g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    if T > 1:
        fl = _flow_stats(g8)
        f["flow_mag_mean"] = float(fl["mag"].mean())
        f["flow_mag_temporal_cv"] = float(fl["mag"].std() / (fl["mag"].mean() + 1e-9))
        f["flow_spatial_std_mean"] = float(fl["spatial"].mean())
        f["flow_vec_std"] = float(fl["vec"].std(axis=0).mean())
        f["flow_coherence"] = float(np.linalg.norm(fl["vec"].mean(axis=0)) / (fl["mag"].mean() + 1e-9))
        f["warp_error_mean"] = float(fl["err"].mean())
        f["warp_error_std"] = float(fl["err"].std())
    else:
        for k in (
            "flow_mag_mean",
            "flow_mag_temporal_cv",
            "flow_spatial_std_mean",
            "flow_vec_std",
            "flow_coherence",
            "warp_error_mean",
            "warp_error_std",
        ):
            f[k] = 0.0
    return f
