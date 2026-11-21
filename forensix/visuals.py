"""Matplotlib figure builders shared by the EDA stage, the reports and the Streamlit Evidence / Compare modules."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .features.image import dct_log_map, fft_log_spectrum, noise_residual, to_gray  # noqa: E402
from .features.video import _flow_stats  # noqa: E402
from .preprocess import image as pi  # noqa: E402
from .preprocess import video as pv  # noqa: E402

REAL_C, FAKE_C = "#2a9d8f", "#e4572e"
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def _new(w=6.0, h=3.6, **kw):
    return plt.subplots(figsize=(w, h), **kw)


# ----------------------------------------------------------------------- dataset-level (EDA)
def hist_by_class(df: pd.DataFrame, feature: str, title: str | None = None):
    fig, ax = _new(5, 3.2)
    for lab, name, c in ((0, "authentic", REAL_C), (1, "manipulated", FAKE_C)):
        ax.hist(df.loc[df["label"] == lab, feature], bins=25, alpha=0.55, color=c, label=name, density=True)
    ax.set_title(title or feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    return fig


def box_by_class(df: pd.DataFrame, features: list[str], title: str = "Feature distributions (z-scored)"):
    z = (df[features] - df[features].mean()) / (df[features].std() + 1e-12)
    fig, ax = _new(max(6, 0.55 * len(features)), 4)
    pos = np.arange(len(features))
    for off, lab, c in ((-0.18, 0, REAL_C), (0.18, 1, FAKE_C)):
        data = [z.loc[df["label"] == lab, f].to_numpy() for f in features]
        bp = ax.boxplot(data, positions=pos + off, widths=0.3, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set(facecolor=c, alpha=0.6, edgecolor="none")
    ax.set_xticks(pos)
    ax.set_xticklabels(features, rotation=60, ha="right")
    ax.set_title(title)
    ax.plot([], [], color=REAL_C, lw=6, label="authentic")
    ax.plot([], [], color=FAKE_C, lw=6, label="manipulated")
    ax.legend(frameon=False, ncol=2)
    return fig


def correlation_heatmap(corr: pd.DataFrame, max_features: int = 30, title: str = "Spearman correlation"):
    if corr.shape[0] > max_features:
        keep = corr.abs().sum().sort_values(ascending=False).index[:max_features]
        corr = corr.loc[keep, keep]
    fig, ax = _new(7, 6)
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title)
    return fig


def effect_size_bars(tests: pd.DataFrame, top: int = 15):
    t = tests.head(top).iloc[::-1]
    fig, ax = _new(6, 0.32 * len(t) + 1)
    ax.barh(t["feature"], t["cohens_d"], color=[FAKE_C if d > 0 else REAL_C for d in t["cohens_d"]])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Cohen's d  (positive = higher in manipulated)")
    ax.set_title("Largest real-vs-manipulated effect sizes")
    return fig


def mean_spectrum_by_class(images_by_class: dict[int, list[np.ndarray]]):
    """Average FFT log-spectrum of authentic vs manipulated images (reveals periodic generator fingerprints)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    specs = {}
    for lab, imgs in images_by_class.items():
        if imgs:
            specs[lab] = np.mean([fft_log_spectrum(to_gray(i)) for i in imgs], axis=0)
    for ax, lab, name in ((axes[0], 0, "authentic"), (axes[1], 1, "manipulated")):
        if lab in specs:
            ax.imshow(specs[lab], cmap="magma")
        ax.set_title(f"mean FFT: {name}")
        ax.axis("off")
    if 0 in specs and 1 in specs:
        d = specs[1] - specs[0]
        v = np.abs(d).max() + 1e-9
        axes[2].imshow(d, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[2].set_title("difference (fake - real)")
    axes[2].axis("off")
    return fig


def sample_grid(samples: list[tuple[np.ndarray, str]], ncols: int = 4, title: str = ""):
    n = len(samples)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (img, cap) in zip(np.atleast_1d(axes).ravel(), samples):
        ax.imshow(img)
        ax.set_title(cap, fontsize=8)
    fig.suptitle(title)
    return fig


# ----------------------------------------------------------------------- image evidence
def image_evidence_figs(rgb_uint8: np.ndarray, size: int = 256) -> dict[str, plt.Figure]:
    img = pi.preprocess_image(rgb_uint8, size)
    gray = to_gray(img)
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(img)
    axes[0].set_title("analysed image")
    axes[1].imshow(fft_log_spectrum(gray), cmap="magma")
    axes[1].set_title("FFT log-spectrum\n(bright dots = periodic artefacts)")
    axes[2].imshow(dct_log_map(gray)[:96, :96], cmap="viridis")
    axes[2].set_title("DCT log-magnitude\n(low frequencies top-left)")
    for a in axes:
        a.axis("off")
    out["frequency"] = fig
    res = noise_residual(gray)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    v = np.percentile(np.abs(res), 99) + 1e-9
    axes[0].imshow(res, cmap="gray", vmin=-v, vmax=v)
    axes[0].set_title("noise residual (image minus smoothed)")
    axes[0].axis("off")
    axes[1].hist(res.ravel(), bins=80, color="#555", density=True)
    axes[1].set_yscale("log")
    axes[1].set_title("residual distribution")
    out["noise"] = fig
    return out


# ----------------------------------------------------------------------- audio evidence
def audio_evidence_figs(y: np.ndarray, sr: int) -> dict[str, plt.Figure]:
    import librosa
    import librosa.display

    out = {}
    fig, ax = _new(8, 2.4)
    t = np.arange(len(y)) / sr
    ax.plot(t, y, lw=0.4, color="#333")
    ax.set_title("waveform")
    ax.set_xlabel("seconds")
    out["waveform"] = fig
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=axes[0], cmap="magma")
    axes[0].set_title("spectrogram (dB)")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    im = axes[1].imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm")
    axes[1].set_title("MFCC heatmap")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("coefficient")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    out["spectral"] = fig
    return out


# ----------------------------------------------------------------------- video evidence
def video_evidence_figs(frames: list[np.ndarray], cfg) -> dict[str, plt.Figure]:
    v = cfg["video"]
    arr = pv.sample_frames(frames, v["frame_interval"], v["max_frames"], v["size"])
    out = {}
    k = min(6, len(arr))
    idx = np.linspace(0, len(arr) - 1, k).astype(int)
    out["frames"] = sample_grid([(arr[i], f"sample {i}") for i in idx], ncols=min(6, k), title="sampled frames")
    gray = np.stack([to_gray(f) for f in arr])
    diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    bright = gray.mean(axis=(1, 2))
    g8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    fl = _flow_stats(g8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0))
    axes[0].plot(diffs, color="#444")
    mu, sd = diffs.mean(), diffs.std()
    spikes = np.where(diffs > mu + 2 * sd)[0]
    axes[0].scatter(spikes, diffs[spikes], color=FAKE_C, zorder=3, label="spike")
    axes[0].set_title("frame-to-frame difference")
    axes[0].legend(frameon=False)
    axes[1].plot(bright, color="#444")
    axes[1].set_title("average brightness")
    axes[2].plot(fl["err"], color="#444", label="warp error")
    axes[2].plot(
        fl["mag"] / (fl["mag"].max() + 1e-9) * fl["err"].max(), color=REAL_C, alpha=0.7, label="motion (scaled)"
    )
    axes[2].set_title("motion-compensated error")
    axes[2].legend(frameon=False)
    for a in axes:
        a.set_xlabel("sampled frame")
    out["temporal"] = fig
    return out


# ----------------------------------------------------------------------- compare
def difference_heatmap(a_rgb: np.ndarray, b_rgb: np.ndarray, size: int = 256):
    """Pixel-wise absolute difference between two versions of the same image + SSIM-like summary numbers."""
    a, b = pi.preprocess_image(a_rgb, size), pi.preprocess_image(b_rgb, size)
    diff = np.abs(a - b).mean(axis=2)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(a)
    axes[0].set_title("version A")
    axes[1].imshow(b)
    axes[1].set_title("version B")
    im = axes[2].imshow(diff, cmap="inferno", vmin=0, vmax=max(diff.max(), 1e-3))
    axes[2].set_title("absolute difference")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    mse = float(((a - b) ** 2).mean())
    psnr = float(10 * np.log10(1.0 / (mse + 1e-12)))
    from skimage.metrics import structural_similarity

    ssim = float(structural_similarity(to_gray(a), to_gray(b), data_range=1.0))
    return fig, {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": ssim,
        "mean_abs_diff": float(diff.mean()),
        "changed_area_pct": float((diff > 0.05).mean() * 100),
    }


def waveform_compare(ya: np.ndarray, yb: np.ndarray, sr: int):
    import librosa

    n = min(len(ya), len(yb))
    ya, yb = ya[:n], yb[:n]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    t = np.arange(n) / sr
    axes[0].plot(t, ya, lw=0.4, label="A", color=REAL_C)
    axes[0].plot(t, yb, lw=0.4, label="B", color=FAKE_C, alpha=0.8)
    axes[0].legend(frameon=False)
    axes[0].set_title("waveforms")
    Sa = librosa.amplitude_to_db(np.abs(librosa.stft(ya, n_fft=512)), ref=1.0)
    Sb = librosa.amplitude_to_db(np.abs(librosa.stft(yb, n_fft=512)), ref=1.0)
    d = Sb - Sa
    v = np.percentile(np.abs(d), 98) + 1e-9
    im = axes[1].imshow(d, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    axes[1].set_title("spectrogram difference (B - A, dB)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    corr = float(np.corrcoef(ya, yb)[0, 1]) if n > 1 else 0.0
    snr = float(10 * np.log10((np.sum(ya**2) + 1e-12) / (np.sum((ya - yb) ** 2) + 1e-12)))
    return fig, {"waveform_correlation": corr, "snr_db": snr, "samples_compared": n}


def video_compare(fa: list[np.ndarray], fb: list[np.ndarray], size: int = 128):
    n = min(len(fa), len(fb))
    per = []
    for i in range(n):
        a, b = pi.preprocess_image(fa[i], size), pi.preprocess_image(fb[i], size)
        per.append(float(np.abs(a - b).mean()))
    fig, ax = _new(7, 2.8)
    ax.plot(per, color="#444")
    ax.set_title("per-frame mean absolute difference (A vs B)")
    ax.set_xlabel("frame")
    j = int(np.argmax(per)) if per else 0
    return fig, {
        "frames_compared": n,
        "mean_abs_diff": float(np.mean(per)) if per else 0.0,
        "max_abs_diff": float(np.max(per)) if per else 0.0,
        "most_different_frame": j,
    }


# ----------------------------------------------------------------------- explanation / benchmarks
def shap_bar(contribs: list[dict], title: str = "What pushed this prediction"):
    items = list(reversed(contribs))
    fig, ax = _new(6.5, 0.38 * len(items) + 1)
    ax.barh(
        [i["feature"] for i in items],
        [i["shap"] for i in items],
        color=[FAKE_C if i["shap"] > 0 else REAL_C for i in items],
    )
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("SHAP value  (right = towards 'manipulated', left = towards 'authentic')")
    ax.set_title(title)
    return fig


def shap_global_bar(importance: pd.Series, top: int = 15):
    t = importance.head(top).iloc[::-1]
    fig, ax = _new(6, 0.32 * len(t) + 1)
    ax.barh(t.index, t.values, color="#3a6ea5")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance")
    return fig


def shap_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, top: int = 12):
    imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-imp)[:top][::-1]
    fig, ax = _new(7, 0.42 * len(order) + 1.2)
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        x = X.iloc[:, j].to_numpy(float)
        rank = (pd.Series(x).rank(pct=True)).to_numpy()
        ax.scatter(shap_values[:, j], row + rng.normal(0, 0.08, len(x)), c=rank, cmap="coolwarm", s=9, alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(X.columns[order])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("SHAP value (colour: feature value low -> high)")
    ax.set_title("SHAP summary")
    return fig


def confusion_fig(cm, title: str = "Confusion matrix"):
    fig, ax = _new(3.4, 3.2)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center", color="black", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["authentic", "manipulated"])
    ax.set_yticklabels(["authentic", "manipulated"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)
    return fig


def roc_pr_fig(results: dict[str, dict], title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, r in results.items():
        t = r["test"]
        if "roc_curve" in t:
            axes[0].plot(t["roc_curve"]["fpr"], t["roc_curve"]["tpr"], label=f"{name} (AUC {t['roc_auc']:.3f})")
            axes[1].plot(
                t["pr_curve"]["recall"], t["pr_curve"]["precision"], label=f"{name} (AP {t['avg_precision']:.3f})"
            )
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.6)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall")
    for a in axes:
        a.legend(frameon=False, fontsize=7)
    fig.suptitle(title)
    return fig


def robustness_bar(df: pd.DataFrame, title: str = "Clean vs degraded F1"):
    d = df.copy()
    fig, ax = _new(8, 0.34 * len(d) + 1.2)
    ax.barh(
        d["transformation"][::-1],
        d["f1"][::-1],
        color=["#3a6ea5" if t == "Clean" else FAKE_C for t in d["transformation"][::-1]],
    )
    ax.axvline(float(d.loc[d["transformation"] == "Clean", "f1"].iloc[0]), color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("F1 (test split)")
    ax.set_title(title)
    return fig


def generator_bar(df: pd.DataFrame, title: str = "Detection rate by generator"):
    d = df.sort_values("detection_rate")
    fig, ax = _new(7, 0.4 * len(d) + 1.2)
    err = np.vstack([d["detection_rate"] - d["ci95_low"], d["ci95_high"] - d["detection_rate"]])
    ax.barh(
        d["source"],
        d["detection_rate"],
        xerr=err,
        color=["#bbb" if s else "#3a6ea5" for s in d["low_support"]],
        ecolor="#555",
        capsize=2,
    )
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of fake test samples detected (95% Wilson CI; grey = low support)")
    ax.set_title(title)
    return fig


def drift_bar(psi: dict, threshold: float, top: int = 12):
    s = pd.Series(psi).sort_values(ascending=False).head(top).iloc[::-1]
    fig, ax = _new(6, 0.32 * len(s) + 1.2)
    ax.barh(s.index, s.values, color=[FAKE_C if v > threshold else "#3a6ea5" for v in s.values])
    ax.axvline(threshold, color="k", ls="--", lw=0.8)
    ax.set_xlabel("PSI vs training baseline")
    ax.set_title("Input feature drift")
    return fig
