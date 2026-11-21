"""Source / generator-wise consistency analysis.

The generator label is used ONLY to evaluate consistency and generalisation of the detector. ForensiX does not try
to identify which generator produced a sample.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ..data.dataset import load_feature_table
from ..mlops.registry import ModelRegistry
from ..utils import get_logger, read_json, write_json

log = get_logger()


def run_generator_analysis(cfg, modality: str) -> pd.DataFrame:
    """Per-generator detection rate on the TEST split, plus a per-generator F1 against all real test samples.

    * ``detection_rate`` (recall of fake) is the fair per-source number: a generator has no real samples of its own.
    * ``f1_vs_real`` pairs a generator's fakes with the full set of real test samples, so F1 values are comparable.
    """
    reg = ModelRegistry(cfg)
    prod = reg.production(modality)
    if prod is None:
        raise RuntimeError(f"no production model for {modality}")
    pdir = reg.model_dir(prod)
    model, meta = joblib.load(pdir / "model.joblib"), read_json(pdir / "meta.json")
    df = load_feature_table(cfg, modality)
    test = df[df["split"] == "test"].copy()
    test["p_fake"] = model.predict_proba(test[meta["features"]].to_numpy(float))[:, 1]
    test["pred"] = (test["p_fake"] >= 0.5).astype(int)
    real = test[test["label"] == 0]
    fake = test[test["label"] == 1]
    far = float((real["pred"] == 1).mean()) if len(real) else np.nan
    rows = []
    min_n = cfg["generators"]["min_support"]
    for src, g in fake.groupby("source"):
        tp, fn = int((g["pred"] == 1).sum()), int((g["pred"] == 0).sum())
        fp = int((real["pred"] == 1).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # Wilson 95% interval on the detection rate, so small groups are not over-read
        n = len(g)
        z = 1.96
        centre = (rec + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(rec * (1 - rec) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append(
            {
                "modality": modality,
                "source": src,
                "n_fake_test": n,
                "detection_rate": rec,
                "ci95_low": max(0.0, centre - half),
                "ci95_high": min(1.0, centre + half),
                "f1_vs_real": f1,
                "mean_p_fake": float(g["p_fake"].mean()),
                "low_support": n < min_n,
            }
        )
    out = pd.DataFrame(rows).sort_values("detection_rate")
    if len(out):
        out["gap_to_best"] = out["detection_rate"].max() - out["detection_rate"]
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.reports_dir / f"generators_{modality}.csv", index=False)
    write_json(
        cfg.reports_dir / f"generators_{modality}.json",
        {
            "modality": modality,
            "model": prod["display"],
            "false_alarm_rate_on_real": far,
            "n_real_test": int(len(real)),
            "spread": float(out["detection_rate"].max() - out["detection_rate"].min()) if len(out) else None,
            "weakest_source": out.iloc[0]["source"] if len(out) else None,
            "strongest_source": out.iloc[-1]["source"] if len(out) else None,
            "fairness_note": "Large spreads indicate the detector generalises unevenly across generators; the gap is "
            "reported, not hidden. Sources below the minimum support carry wide intervals.",
        },
    )
    if len(out):
        log.info(
            "[%s] detection rate by source: %s",
            modality,
            ", ".join(f"{r.source}={r.detection_rate:.2f}(n={r.n_fake_test})" for r in out.itertuples()),
        )
    return out
