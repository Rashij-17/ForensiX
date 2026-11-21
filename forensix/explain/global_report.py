"""Stage 7: global SHAP analysis of the production model + evaluation figures (confusion, ROC, PR)."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from ..data.dataset import load_feature_table
from ..mlops.registry import ModelRegistry
from ..utils import get_logger, read_json, write_json
from .. import visuals as V
from .shap_explain import explain_global

log = get_logger()


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_global_explanations(cfg, modality: str) -> dict:
    reg = ModelRegistry(cfg)
    prod = reg.production(modality)
    if prod is None:
        raise RuntimeError(f"no production model for {modality}")
    pdir = reg.model_dir(prod)
    model, meta = joblib.load(pdir / "model.joblib"), read_json(pdir / "meta.json")
    bg = pd.read_csv(pdir / "background.csv")
    df = load_feature_table(cfg, modality)
    test = df[df["split"] == "test"]
    X = test[meta["features"]]
    g = explain_global(model, prod["algorithm"], bg, X, cfg["shap"]["max_explain_samples"], cfg["seed"])
    out = cfg.reports_dir / "shap" / modality
    _save(V.shap_global_bar(g["importance"]), out / "importance.png")
    _save(V.shap_beeswarm(g["shap_values"], g["X"]), out / "beeswarm.png")
    g["importance"].rename("mean_abs_shap").to_csv(out / "importance.csv")
    g["groups"].rename("mean_abs_shap").to_csv(out / "indicator_groups.csv")

    rep = json.loads((cfg.reports_dir / f"models_{modality}.json").read_text())
    _save(
        V.roc_pr_fig(rep["results"], f"{modality}: all four models (test split)"),
        cfg.reports_dir / "eval" / f"{modality}_roc_pr.png",
    )
    best = rep["results"][prod["algorithm"]]["test"]
    _save(
        V.confusion_fig(best["confusion"], f"{prod['display']} (test)"),
        cfg.reports_dir / "eval" / f"{modality}_confusion.png",
    )
    summary = {
        "modality": modality,
        "model": prod["display"],
        "top_features": g["importance"].head(10).round(5).to_dict(),
        "indicator_groups": g["groups"].round(5).to_dict(),
        "n_explained": int(len(g["X"])),
    }
    write_json(out / "summary.json", summary)
    log.info("[%s] SHAP top features: %s", modality, ", ".join(list(summary["top_features"])[:5]))
    return summary
