"""Human-readable descriptions and indicator groups for every engineered feature (used by SHAP reports)."""

from __future__ import annotations

import re

_EXACT: dict[str, tuple[str, str]] = {
    "gray_var": ("Statistical", "variance of grey-level intensity"),
    "gray_skew": ("Statistical", "skewness of the intensity distribution"),
    "gray_kurt": ("Statistical", "kurtosis (tail weight) of the intensity distribution"),
    "gray_entropy": ("Statistical", "Shannon entropy of intensity (information content)"),
    "channel_corr_rg": ("Statistical", "red-green channel correlation"),
    "channel_corr_gb": ("Statistical", "green-blue channel correlation"),
    "edge_density": ("Texture", "fraction of pixels on detected edges"),
    "local_var_mean": ("Texture", "mean local variance (fine texture energy)"),
    "local_var_std": ("Texture", "spread of local variance across the image"),
    "laplacian_var": ("Texture", "Laplacian variance (sharpness / fine detail)"),
    "gradient_mean": ("Texture", "mean gradient magnitude"),
    "noise_std": ("Noise", "standard deviation of the noise residual"),
    "noise_kurt": ("Noise", "kurtosis of the noise residual"),
    "noise_abs_mean": ("Noise", "mean absolute noise residual"),
    "noise_autocorr_h1": ("Noise", "horizontal 1-pixel noise autocorrelation (resampling / upsampling trace)"),
    "noise_autocorr_v1": ("Noise", "vertical 1-pixel noise autocorrelation"),
    "noise_autocorr_h2": ("Noise", "horizontal 2-pixel noise autocorrelation"),
    "fft_low_ratio": ("Frequency", "share of spectral energy at low spatial frequencies"),
    "fft_mid_ratio": ("Frequency", "share of spectral energy at mid spatial frequencies"),
    "fft_hf_ratio": ("Frequency", "share of spectral energy at high spatial frequencies"),
    "fft_centroid": ("Frequency", "spectral centroid of the 2-D FFT"),
    "fft_slope": ("Frequency", "slope of the radial power spectrum (1/f falloff)"),
    "fft_hf_peak_z": ("Frequency", "strength of isolated high-frequency spectral peaks (periodic artefacts)"),
    "fft_hf_axis_ratio": ("Frequency", "high-frequency energy concentrated along the horizontal/vertical axes"),
    "dct_hf_ratio": ("Frequency", "share of DCT energy in high-frequency coefficients"),
    "dct_ac_mean": ("Frequency", "mean DCT AC coefficient magnitude"),
    "dct_ac_std": ("Frequency", "spread of DCT AC coefficient magnitude"),
    "blockiness": ("Frequency", "8x8 block-boundary discontinuity (compression history)"),
    "mfcc_delta_std_mean": ("Spectral", "variability of MFCC change over time"),
    "spec_flatness_mean": ("Spectral", "spectral flatness (noisiness vs tonality)"),
    "spec_contrast_mean": ("Spectral", "spectral peak-to-valley contrast"),
    "spec_flux_std": ("Spectral", "irregularity of frame-to-frame spectral change"),
    "rms_dyn_range_db": ("Energy", "dynamic range of loudness (dB)"),
    "noise_floor_db": ("Energy", "background noise floor level (dB)"),
    "hf_energy_ratio_4k": ("Spectral", "share of energy above 4 kHz"),
    "hf_energy_ratio_7k": ("Spectral", "share of energy above 7 kHz"),
    "pitch_mean": ("Pitch", "mean fundamental frequency"),
    "pitch_std": ("Pitch", "fundamental-frequency variation"),
    "pitch_jitter": ("Pitch", "frame-to-frame pitch jitter"),
    "pitch_plausible_frac": ("Pitch", "fraction of frames with a plausible voiced pitch"),
    "brightness_std": ("Temporal", "variation of average brightness across frames"),
    "brightness_change_mean": ("Temporal", "average frame-to-frame brightness change"),
    "brightness_change_max": ("Temporal", "largest frame-to-frame brightness jump"),
    "diff_mean": ("Temporal", "mean absolute frame-to-frame difference"),
    "diff_std": ("Temporal", "variability of frame differences"),
    "diff_max": ("Temporal", "largest frame-to-frame difference"),
    "diff_cv": ("Temporal", "relative variability of frame differences"),
    "diff_jerk": ("Temporal", "abruptness of change in frame differences"),
    "texture_change": ("Temporal", "frame-to-frame change in texture sharpness"),
    "noise_temporal_corr": ("Temporal", "correlation of noise residual between consecutive frames"),
    "flow_mag_mean": ("Motion", "average optical-flow motion magnitude"),
    "flow_mag_temporal_cv": ("Motion", "motion magnitude variability over time"),
    "flow_spatial_std_mean": ("Motion", "spatial variability of motion within frames"),
    "flow_vec_std": ("Motion", "variability of the global motion direction"),
    "flow_coherence": ("Motion", "coherence of global motion"),
    "warp_error_mean": ("Motion", "motion-compensated prediction error (temporal inconsistency)"),
    "warp_error_std": ("Motion", "variability of motion-compensated prediction error"),
}

_PREFIX: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^mean_([rgb])$"), "Statistical", r"mean \1-channel intensity"),
    (re.compile(r"^std_([rgb])$"), "Statistical", r"\1-channel contrast (std)"),
    (re.compile(r"^mfcc(\d+)_(mean|std)$"), "Spectral", r"MFCC \1 (\2)"),
    (re.compile(r"^spec_(centroid|bandwidth|rolloff)_(mean|std)$"), "Spectral", r"spectral \1 (\2)"),
    (re.compile(r"^zcr_(mean|std)$"), "Spectral", r"zero-crossing rate (\1)"),
    (re.compile(r"^rms_(mean|std)$"), "Energy", r"RMS energy (\1)"),
]


def describe_feature(name: str) -> tuple[str, str]:
    """Return (indicator group, plain-language description) for a feature name."""
    if name in _EXACT:
        return _EXACT[name]
    for pat, grp, tmpl in _PREFIX:
        m = pat.match(name)
        if m:
            return grp, m.expand(tmpl)
    m = re.match(r"^frame_(.+)_(mean|std)$", name)
    if m:
        grp, desc = describe_feature(m.group(1))
        how = "average over frames" if m.group(2) == "mean" else "variation across frames"
        return grp + " (per-frame)", f"{desc}, {how}"
    return "Other", name.replace("_", " ")


def indicator_group(name: str) -> str:
    return describe_feature(name)[0]
