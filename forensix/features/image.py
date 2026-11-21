"""Image forensic features: statistical, spatial/texture, noise-residual, FFT and DCT measures."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from scipy import stats as sps


def to_gray(img: np.ndarray) -> np.ndarray:
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


@lru_cache(maxsize=8)
def _radial_maps(h: int, w: int):
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    dy, dx = yy - cy, xx - cx
    r = np.hypot(dy, dx)
    r_int = np.minimum(r.astype(np.int32), min(h, w) // 2 - 1)
    angle = np.degrees(np.arctan2(dy, dx)) % 180.0
    return r, r_int, angle, float(min(h, w) // 2)


def entropy_of(values: np.ndarray, bins: int = 256, value_range=(0.0, 1.0)) -> float:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def fft_log_spectrum(gray: np.ndarray) -> np.ndarray:
    """Centred log-magnitude 2-D spectrum (also used for the Evidence visualisation)."""
    win = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1])).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win))
    return np.log1p(np.abs(f))


def noise_residual(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gray - cv2.GaussianBlur(gray, (0, 0), sigma)


def dct_log_map(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    g = gray[: h - h % 2, : w - w % 2]
    return np.log1p(np.abs(cv2.dct(g.astype(np.float32))))


def extract_image_features(img: np.ndarray) -> dict[str, float]:
    """``img`` is float32 RGB in [0, 1] at the common resolution. Returns an ordered feature dict."""
    f: dict[str, float] = {}
    gray = to_gray(img)
    h, w = gray.shape

    # ---- statistical -------------------------------------------------
    for i, c in enumerate("rgb"):
        f[f"mean_{c}"] = float(img[..., i].mean())
        f[f"std_{c}"] = float(img[..., i].std())
    g = gray.ravel()
    f["gray_var"] = float(g.var())
    f["gray_skew"] = float(sps.skew(g))
    f["gray_kurt"] = float(sps.kurtosis(g))
    f["gray_entropy"] = entropy_of(g)
    f["channel_corr_rg"] = float(np.corrcoef(img[..., 0].ravel(), img[..., 1].ravel())[0, 1]) if g.std() > 0 else 0.0
    f["channel_corr_gb"] = float(np.corrcoef(img[..., 1].ravel(), img[..., 2].ravel())[0, 1]) if g.std() > 0 else 0.0

    # ---- spatial / texture -------------------------------------------
    gray8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    f["edge_density"] = float((cv2.Canny(gray8, 100, 200) > 0).mean())
    mean_loc = cv2.blur(gray, (5, 5))
    local_var = np.maximum(cv2.blur(gray * gray, (5, 5)) - mean_loc * mean_loc, 0.0)
    f["local_var_mean"] = float(local_var.mean())
    f["local_var_std"] = float(local_var.std())
    f["laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    f["gradient_mean"] = float(np.hypot(gx, gy).mean())

    # ---- noise residual ----------------------------------------------
    res = noise_residual(gray)
    f["noise_std"] = float(res.std())
    f["noise_kurt"] = float(sps.kurtosis(res.ravel()))
    f["noise_abs_mean"] = float(np.abs(res).mean())
    r0 = res - res.mean()
    denom = float((r0 * r0).mean()) + 1e-12
    f["noise_autocorr_h1"] = float((r0[:, :-1] * r0[:, 1:]).mean() / denom)
    f["noise_autocorr_v1"] = float((r0[:-1, :] * r0[1:, :]).mean() / denom)
    f["noise_autocorr_h2"] = float((r0[:, :-2] * r0[:, 2:]).mean() / denom)

    # ---- FFT ----------------------------------------------------------
    r, r_int, angle, rmax = _radial_maps(h, w)
    logspec = fft_log_spectrum(gray)
    power = np.expm1(logspec) ** 2
    total = float(power.sum()) + 1e-12
    f["fft_low_ratio"] = float(power[r < 0.25 * rmax].sum() / total)
    f["fft_mid_ratio"] = float(power[(r >= 0.25 * rmax) & (r < 0.5 * rmax)].sum() / total)
    f["fft_hf_ratio"] = float(power[(r >= 0.5 * rmax) & (r <= rmax)].sum() / total)
    f["fft_centroid"] = float((r * power)[r <= rmax].sum() / total / rmax)
    prof = np.bincount(r_int.ravel(), weights=power.ravel()) / np.maximum(np.bincount(r_int.ravel()), 1)
    xs = np.arange(len(prof))
    sel = (xs >= 2) & (xs <= 0.9 * rmax) & (prof > 0)
    slope = np.polyfit(np.log(xs[sel]), np.log(prof[sel]), 1)[0] if sel.sum() > 3 else 0.0
    f["fft_slope"] = float(slope)
    hf = (r >= 0.5 * rmax) & (r <= rmax)
    hf_vals = logspec[hf]
    f["fft_hf_peak_z"] = float((hf_vals.max() - hf_vals.mean()) / (hf_vals.std() + 1e-9))
    axis_band = (np.minimum(angle, 180 - angle) < 12) | (np.abs(angle - 90) < 12)
    f["fft_hf_axis_ratio"] = float(power[hf & axis_band].sum() / (power[hf].sum() + 1e-12))

    # ---- DCT ----------------------------------------------------------
    d = cv2.dct(gray[: h - h % 2, : w - w % 2].astype(np.float32))
    uu, vv = np.indices(d.shape)
    d2 = d * d
    dsum = float(d2.sum()) + 1e-12
    f["dct_hf_ratio"] = float(d2[(uu + vv) > 0.5 * (d.shape[0] + d.shape[1])].sum() / dsum)
    ac = np.abs(d).ravel()[1:]
    f["dct_ac_mean"] = float(ac.mean())
    f["dct_ac_std"] = float(ac.std())
    col_diff = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    idx = np.arange(col_diff.size)
    on_edge = col_diff[(idx % 8) == 7].mean() if (idx % 8 == 7).any() else 0.0
    off_edge = col_diff[(idx % 8) != 7].mean() if (idx % 8 != 7).any() else 1.0
    f["blockiness"] = float(on_edge / (off_edge + 1e-9))
    return f
