"""Configuration loading and project paths."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

MODALITIES = ("image", "audio", "video")
MODEL_NAMES = ("logistic_regression", "svm", "random_forest", "xgboost")
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "versions": {"preprocessing": "pp-1", "feature": "ft-1"},
    "data": {"raw_dir": "data/raw", "splits": {"train": 0.70, "val": 0.15, "test": 0.15}},
    "image": {"size": 256},
    "audio": {"sr": 16000, "segment_seconds": 4.0, "max_segments": 3, "min_seconds": 0.5, "silence_db": -55.0},
    "video": {
        "size": 128,
        "frame_interval": 2,
        "max_frames": 24,
        "min_frames": 6,
        "max_raw_frames": 120,
        "max_side": 320,
    },
    "models": {
        "logistic_regression": {"C": 1.0, "max_iter": 3000},
        "svm": {"C": 3.0, "kernel": "rbf", "gamma": "scale"},
        "random_forest": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
    },
    "training": {"selection_metric": "f1", "cv_folds": 5},
    "decision": {"authentic_below": 0.35, "manipulated_above": 0.65},
    "shap": {"max_explain_samples": 150, "background_samples": 25, "top_k": 8},
    "robustness": {
        "image": {"jpeg": [70, 30], "resize": [0.75, 0.5], "blur": [1.0, 2.0], "noise": [4, 10]},
        "audio": {"bgnoise": [20, 10], "mp3": [64, 32], "resample": [11025, 8000]},
        "video": {"h264": [28, 38], "downscale": [0.75, 0.5], "fps_div": [2, 3]},
    },
    "generators": {"min_support": 5},
    "monitoring": {
        "window": 500,
        "min_samples": 30,
        "psi_threshold": 0.2,
        "drift_share_trigger": 0.30,
        "f1_drop_trigger": 0.05,
        "min_labeled": 20,
        "min_improvement": 0.0,
    },
    "privacy": {"retain_uploads": False, "max_upload_mb": 200},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Wraps the parameter dictionary and resolves all project paths."""

    def __init__(self, params: dict, root: Path):
        self.p = params
        self.root = Path(root)

    # ---- parameters -------------------------------------------------
    def get(self, *keys: str, default: Any = None) -> Any:
        cur: Any = self.p
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def __getitem__(self, key: str) -> Any:
        return self.p[key]

    # ---- paths ------------------------------------------------------
    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def raw_dir(self) -> Path:
        return self.root / self.p["data"]["raw_dir"]

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def features_dir(self) -> Path:
        return self.root / "data" / "features"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def monitoring_dir(self) -> Path:
        return self.artifacts_dir / "monitoring"

    @property
    def metadata_path(self) -> Path:
        return self.processed_dir / "metadata.csv"

    def features_path(self, modality: str) -> Path:
        return self.features_dir / f"{modality}_features.csv"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.processed_dir,
            self.features_dir,
            self.models_dir,
            self.reports_dir,
            self.monitoring_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def get_config(root: str | os.PathLike | None = None) -> Config:
    """Load params.yaml from the project root (or FORENSIX_HOME) merged over the defaults."""
    root_path = Path(root or os.environ.get("FORENSIX_HOME") or Path(__file__).resolve().parents[1])
    params = copy.deepcopy(DEFAULTS)
    pfile = next((f for f in (root_path / "config" / "params.yaml", root_path / "params.yaml") if f.exists()), None)
    if pfile is not None:
        with open(pfile, "r", encoding="utf-8") as fh:
            params = _merge(params, yaml.safe_load(fh) or {})
    cfg = Config(params, root_path)
    cfg.ensure_dirs()
    return cfg
