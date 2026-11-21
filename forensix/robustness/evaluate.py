"""Robustness benchmark: clean F1 vs degraded F1 on the TEST split for each transformation."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, recall_score

from ..data.dataset import load_metadata
from ..features import features_from_frames, features_from_image_array, features_from_waveform, load_media
from ..mlops.registry import ModelRegistry
from ..utils import get_logger, read_json
from . import transforms as T

log = get_logger()


def _transform_one(path: str, modality: str, cfg, names: list[str]) -> dict[str, dict | None]:
    """Load once, return {transform name -> feature dict} (+ 'Clean')."""
    media = load_media(path, modality, cfg)
    out: dict[str, dict | None] = {}
    if media is None:
        return out
    if modality == "image":
        tf = T.image_transforms(cfg)
        out["Clean"] = features_from_image_array(media, cfg)
        for n in names:
            try:
                out[n] = features_from_image_array(tf[n](media), cfg)
            except Exception:
                out[n] = None
    elif modality == "audio":
        tf = T.audio_transforms(cfg)
        out["Clean"] = features_from_waveform(media, cfg)
        for n in names:
            try:
                out[n] = features_from_waveform(tf[n](media), cfg)
            except Exception:
                out[n] = None
    else:
        frames, fps = media
        tf = T.video_transforms(cfg)
        out["Clean"] = features_from_frames(frames, cfg)
        for n in names:
            try:
                out[n] = features_from_frames(tf[n](frames, fps), cfg)
            except Exception:
                out[n] = None
    return out


def run_robustness(cfg, modality: str, n_jobs: int = -1, max_samples: int | None = None) -> pd.DataFrame:
    """Stage 9: re-score the production model on transformed copies of the test set.

    Samples whose degraded version cannot be analysed (e.g. frame-rate reduction leaving too few frames) are
    reported in the ``n_failed`` column rather than silently dropped from the denominator.
    """
    reg = ModelRegistry(cfg)
    prod = reg.production(modality)
    if prod is None:
        raise RuntimeError(f"no production model for {modality}; train first")
    pdir = reg.model_dir(prod)
    model, meta = joblib.load(pdir / "model.joblib"), read_json(pdir / "meta.json")
    feats = meta["features"]
    md = load_metadata(cfg)
    test = md[(md["modality"] == modality) & (md["split"] == "test") & (md["status"] == "ok")]
    if max_samples:
        test = test.sample(n=min(max_samples, len(test)), random_state=cfg["seed"])
    names = list({"image": T.image_transforms, "audio": T.audio_transforms, "video": T.video_transforms}[modality](cfg))
    log.info("[%s] robustness on %d test samples x %d transforms", modality, len(test), len(names))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_transform_one)(r["abs_path"], modality, cfg, names) for r in test.to_dict("records")
    )
    y_all = test["label"].to_numpy(int)

    def score(key: str):
        idx = [i for i, r in enumerate(res) if r.get(key) is not None]
        n_failed = len(res) - len(idx)
        if len(idx) < 2:
            return None, n_failed
        X = pd.DataFrame([res[i][key] for i in idx])[feats].to_numpy(float)
        y = y_all[idx]
        p = model.predict_proba(X)[:, 1]
        pred = (p >= 0.5).astype(int)
        return {
            "f1": f1_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "recall_real": recall_score(y, pred, pos_label=0, zero_division=0),
            "mean_p_fake": float(p.mean()),
            "n": len(idx),
            "pred": pred,
            "idx": idx,
        }, n_failed

    clean, clean_failed = score("Clean")
    if clean is None:
        raise RuntimeError("clean test set could not be scored")
    rows = [
        {
            "transformation": "Clean",
            "family": "Baseline",
            "f1": clean["f1"],
            "clean_f1": clean["f1"],
            "f1_drop": 0.0,
            "relative_drop": 0.0,
            "recall_fake": clean["recall"],
            "recall_real": clean["recall_real"],
            "mean_p_fake": clean["mean_p_fake"],
            "n_scored": clean["n"],
            "n_failed": clean_failed,
        }
    ]
    for n in names:
        s, failed = score(n)
        if s is None:
            rows.append(
                {
                    "transformation": n,
                    "family": T.transform_family(n),
                    "f1": np.nan,
                    "clean_f1": clean["f1"],
                    "f1_drop": np.nan,
                    "relative_drop": np.nan,
                    "n_scored": 0,
                    "n_failed": failed,
                }
            )
            continue
        # clean F1 recomputed on the SAME surviving subset so the comparison is like for like
        sub = [clean["idx"].index(i) for i in s["idx"] if i in clean["idx"]]
        clean_sub = f1_score(y_all[s["idx"]], clean["pred"][sub], zero_division=0) if sub else clean["f1"]
        rows.append(
            {
                "transformation": n,
                "family": T.transform_family(n),
                "f1": s["f1"],
                "clean_f1": clean_sub,
                "f1_drop": clean_sub - s["f1"],
                "relative_drop": (clean_sub - s["f1"]) / clean_sub if clean_sub else np.nan,
                "recall_fake": s["recall"],
                "recall_real": s["recall_real"],
                "mean_p_fake": s["mean_p_fake"],
                "n_scored": s["n"],
                "n_failed": failed,
            }
        )
    out = pd.DataFrame(rows)
    out.insert(0, "modality", modality)
    out["model"] = prod["display"]
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.reports_dir / f"robustness_{modality}.csv", index=False)
    worst = out[out["transformation"] != "Clean"].sort_values("f1_drop", ascending=False).head(1)
    if len(worst):
        w = worst.iloc[0]
        log.info(
            "[%s] clean F1 %.3f; worst: %s (F1 %.3f, drop %.3f)",
            modality,
            clean["f1"],
            w["transformation"],
            w["f1"],
            w["f1_drop"],
        )
    return out
