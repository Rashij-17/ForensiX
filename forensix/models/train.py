"""Model training, evaluation and selection (Logistic Regression, SVM, Random Forest, XGBoost) per modality."""

from __future__ import annotations

import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ..config import MODEL_LABELS, MODEL_NAMES
from ..data.dataset import feature_columns, load_feature_table
from ..mlops.registry import ModelRegistry
from ..mlops.tracking import tracked_run
from ..mlops.versioning import current_version
from ..utils import get_logger, write_json

log = get_logger()


def make_model(name: str, cfg):
    p = dict(cfg["models"][name])
    seed = cfg["seed"]
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **p))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed, **p))
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    if name == "xgboost":
        return XGBClassifier(random_state=seed, n_jobs=2, eval_metric="logloss", verbosity=0, **p)
    raise ValueError(name)


def _thin(a: np.ndarray, k: int = 200) -> list[float]:
    if len(a) <= k:
        return [float(x) for x in a]
    idx = np.linspace(0, len(a) - 1, k).astype(int)
    return [float(a[i]) for i in idx]


def evaluate_scores(y: np.ndarray, p: np.ndarray, threshold: float = 0.5, curves: bool = True) -> dict:
    """Accuracy, precision, recall, F1, ROC AUC, confusion matrix, ROC and precision-recall curves (fake = positive)."""
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "avg_precision": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if curves and len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        out["roc_curve"] = {"fpr": _thin(fpr), "tpr": _thin(tpr)}
        out["pr_curve"] = {"precision": _thin(pr), "recall": _thin(rc)}
    return out


def _cv_f1(model, X, y, folds: int, seed: int) -> float:
    k = int(min(folds, np.bincount(y).min()))
    if k < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return float(cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1).mean())


def _split(df: pd.DataFrame, feats: list[str]):
    parts = {}
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        parts[s] = (d[feats].to_numpy(float), d["label"].to_numpy(int), d)
    return parts


def _prob_fake(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def reference_stats(train: pd.DataFrame, feats: list[str]) -> dict:
    """Per-class distribution summary of every feature, used to put a single sample in statistical context."""
    out = {}
    for f in feats:
        r, k = train.loc[train["label"] == 0, f].to_numpy(float), train.loc[train["label"] == 1, f].to_numpy(float)
        out[f] = {
            "real": {
                "mean": float(r.mean()),
                "std": float(r.std() + 1e-12),
                "pct": [float(x) for x in np.percentile(r, [5, 25, 50, 75, 95])],
            },
            "fake": {
                "mean": float(k.mean()),
                "std": float(k.std() + 1e-12),
                "pct": [float(x) for x in np.percentile(k, [5, 25, 50, 75, 95])],
            },
        }
    return out


def train_modality(
    cfg,
    modality: str,
    extra_train: pd.DataFrame | None = None,
    register: bool = True,
    promote_first: bool = True,
    note: str = "",
    dataset_version: str | None = None,
) -> dict:
    """Train & compare the four models for one modality.

    * all four are fitted on the TRAIN split (plus optional verified production samples ``extra_train``);
    * the winner is chosen on the VALIDATION split only; the TEST split is reported, never used for selection;
    * every candidate is registered (staging). The winner becomes production only when no production model
      exists yet (initial approval) - later promotions go through the retraining gate.
    """
    df = load_feature_table(cfg, modality)
    feats = feature_columns(df)
    parts = _split(df, feats)
    (Xtr, ytr, dtr), (Xva, yva, _), (Xte, yte, _) = parts["train"], parts["val"], parts["test"]
    if extra_train is not None and len(extra_train):
        Xtr = np.vstack([Xtr, extra_train[feats].to_numpy(float)])
        ytr = np.concatenate([ytr, extra_train["label"].to_numpy(int)])
        dtr = pd.concat([dtr, extra_train.assign(split="train")], ignore_index=True)
    if len(np.unique(ytr)) < 2 or len(yva) == 0 or len(yte) == 0:
        raise RuntimeError(f"[{modality}] need both classes in train and non-empty val/test splits")

    registry = ModelRegistry(cfg)
    cycle = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    ds_version = dataset_version or current_version(cfg)
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    run_ids: dict[str, str | None] = {}

    for name in MODEL_NAMES:
        model = make_model(name, cfg)
        params = {
            "modality": modality,
            "model": name,
            "feature_set": cfg["versions"]["feature"],
            "n_features": len(feats),
            "n_train": len(ytr),
            "dataset_version": ds_version,
            **{f"hp_{k}": v for k, v in cfg["models"][name].items()},
        }
        with tracked_run(cfg, modality, f"{modality}-{name}-{cycle}", params, {"cycle": cycle}) as (log_run, run_id):
            model.fit(Xtr, ytr)
            va = evaluate_scores(yva, _prob_fake(model, Xva))
            te = evaluate_scores(yte, _prob_fake(model, Xte))
            tr = evaluate_scores(ytr, _prob_fake(model, Xtr), curves=False)
            cv = _cv_f1(make_model(name, cfg), Xtr, ytr, cfg["training"]["cv_folds"], cfg["seed"])
            log_run(
                metrics={
                    **{f"val_{k}": va[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
                    **{f"test_{k}": te[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
                    "train_f1": tr["f1"],
                    "cv_f1": cv,
                }
            )
        results[name] = {"train": tr, "val": va, "test": te, "cv_f1": cv}
        fitted[name], run_ids[name] = model, run_id

    metric = cfg["training"]["selection_metric"]
    ranking = sorted(
        MODEL_NAMES,
        key=lambda n: (np.nan_to_num(results[n]["val"][metric]), np.nan_to_num(results[n]["val"]["roc_auc"])),
        reverse=True,
    )
    winner = ranking[0]
    log.info(
        "[%s] validation %s: %s -> winner %s",
        modality,
        metric,
        ", ".join(f"{n}={results[n]['val'][metric]:.3f}" for n in MODEL_NAMES),
        winner,
    )

    background = pd.DataFrame(Xtr, columns=feats).sample(
        n=min(len(Xtr), cfg["shap"]["background_samples"]), random_state=cfg["seed"]
    )
    ref = reference_stats(pd.DataFrame(Xtr, columns=feats).assign(label=ytr), feats)
    entries = {}
    for name in MODEL_NAMES:
        version = registry.next_version(modality, name)
        rel = f"models/{modality}/{name}-v{version}"
        d = cfg.root / rel
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[name], d / "model.joblib")
        background.to_csv(d / "background.csv", index=False)
        write_json(d / "reference_stats.json", ref)
        meta = {
            "modality": modality,
            "algorithm": name,
            "version": version,
            "features": feats,
            "feature_version": cfg["versions"]["feature"],
            "dataset_version": ds_version,
            "cycle": cycle,
            "metrics": {
                s: {k: v for k, v in results[name][s].items() if k not in ("roc_curve", "pr_curve")}
                for s in ("train", "val", "test")
            },
            "cv_f1": results[name]["cv_f1"],
            "n_train": int(len(ytr)),
            "selected_as_winner": name == winner,
            "threshold": 0.5,
            "hyperparameters": cfg["models"][name],
        }
        write_json(d / "meta.json", meta)
        if register:
            summary = {
                "val_f1": results[name]["val"]["f1"],
                "val_roc_auc": results[name]["val"]["roc_auc"],
                "test_accuracy": results[name]["test"]["accuracy"],
                "test_precision": results[name]["test"]["precision"],
                "test_recall": results[name]["test"]["recall"],
                "test_f1": results[name]["test"]["f1"],
                "test_roc_auc": results[name]["test"]["roc_auc"],
            }
            entries[name] = registry.register(
                modality,
                name,
                summary,
                rel,
                version,
                ds_version,
                cfg["versions"]["feature"],
                run_ids[name],
                "staging",
                note or f"trained {cycle}",
                cycle,
            )
        else:
            entries[name] = {
                "modality": modality,
                "algorithm": name,
                "version": version,
                "path": rel,
                "metrics": results[name]["test"],
            }

    promoted = False
    if register and promote_first and registry.production(modality) is None:
        e = entries[winner]
        registry.set_stage(modality, winner, e["version"], "production", "initial approval: best validation F1")
        promoted = True
        from ..mlops.monitoring import save_baseline

        save_baseline(
            cfg,
            modality,
            pd.DataFrame(Xtr, columns=feats),
            _prob_fake(fitted[winner], Xva),
            results[winner]["test"]["f1"],
        )

    report = {
        "modality": modality,
        "cycle": cycle,
        "winner": winner,
        "ranking": ranking,
        "promoted_initial": promoted,
        "dataset_version": ds_version,
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
        "results": results,
        "versions": {n: entries[n].get("version") for n in MODEL_NAMES},
    }
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.reports_dir / f"models_{modality}.json", report)
    return report


def comparison_table(cfg) -> pd.DataFrame:
    """Flat modality x model benchmark table (test metrics, validation F1, CV F1) from the saved reports."""
    rows = []
    for m in ("image", "audio", "video"):
        p = cfg.reports_dir / f"models_{m}.json"
        if not p.exists():
            continue
        rep = json.loads(p.read_text())
        for name, r in rep["results"].items():
            t = r["test"]
            rows.append(
                {
                    "modality": m,
                    "model": MODEL_LABELS[name],
                    "accuracy": t["accuracy"],
                    "precision": t["precision"],
                    "recall": t["recall"],
                    "f1": t["f1"],
                    "roc_auc": t["roc_auc"],
                    "val_f1": r["val"]["f1"],
                    "cv_f1": r["cv_f1"],
                    "n_test": t["n"],
                    "winner": name == rep["winner"],
                }
            )
    return pd.DataFrame(rows)
