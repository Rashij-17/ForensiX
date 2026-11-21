"""Stage 4: exploratory data analysis. Figures go to ``reports/eda/<modality>/`` and a JSON summary."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .data.dataset import feature_columns, load_feature_table, load_metadata
from .features import load_media
from .preprocess import image as pi
from .stats.analysis import correlation_study, feature_tests
from .utils import get_logger, write_json
from . import visuals as V

log = get_logger()


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_eda(cfg, modality: str, n_examples: int = 4) -> dict:
    out = cfg.reports_dir / "eda" / modality
    df = load_feature_table(cfg, modality)
    train = df[df["split"] == "train"]
    feats = feature_columns(df)
    tests = feature_tests(train, feats)
    top = tests.head(8)["feature"].tolist() if not tests.empty else feats[:8]
    for f in top[:6]:
        _save(V.hist_by_class(train, f), out / f"hist_{f}.png")
    _save(V.box_by_class(train, top), out / "boxplots_top_features.png")
    cor = correlation_study(train, feats)
    _save(
        V.correlation_heatmap(cor["corr"], title=f"{modality}: Spearman correlation of features"),
        out / "correlation_matrix.png",
    )
    if not tests.empty:
        _save(V.effect_size_bars(tests), out / "effect_sizes.png")

    meta = load_metadata(cfg)
    ok = meta[(meta["modality"] == modality) & (meta["status"] == "ok") & (meta["split"] == "train")]
    picks = {lab: ok[ok["label"] == lab].head(n_examples) for lab in (0, 1)}
    if modality == "image":
        imgs = {
            lab: [pi.preprocess_image(load_media(r, "image", cfg), cfg.get("image", "size")) for r in p["abs_path"]]
            for lab, p in picks.items()
        }
        _save(V.mean_spectrum_by_class(imgs), out / "mean_fft_real_vs_fake.png")
        samples = [(i, "authentic") for i in imgs[0]] + [(i, "manipulated") for i in imgs[1]]
        _save(V.sample_grid(samples, ncols=n_examples, title="example images"), out / "examples.png")
    elif modality == "audio":
        for lab, name in ((0, "authentic"), (1, "manipulated")):
            if len(picks[lab]):
                y = load_media(picks[lab]["abs_path"].iloc[0], "audio", cfg)
                for k, fig in V.audio_evidence_figs(y, cfg.get("audio", "sr")).items():
                    _save(fig, out / f"{name}_{k}.png")
    else:
        for lab, name in ((0, "authentic"), (1, "manipulated")):
            if len(picks[lab]):
                frames, _ = load_media(picks[lab]["abs_path"].iloc[0], "video", cfg)
                for k, fig in V.video_evidence_figs(frames, cfg).items():
                    _save(fig, out / f"{name}_{k}.png")
    summary = {
        "modality": modality,
        "n_train": int(len(train)),
        "class_balance": train["label"].value_counts().to_dict(),
        "n_features": len(feats),
        "top_features_by_effect": top,
        "figures_dir": str(out),
    }
    write_json(out / "summary.json", summary)
    log.info("[%s] EDA figures written to %s", modality, out)
    return summary
