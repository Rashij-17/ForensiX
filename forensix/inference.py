"""Inference service: media file -> modality -> features -> probability -> verdict -> SHAP explanation."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd

from .config import get_config
from .explain.shap_explain import explain_sample
from .features import extract_from_path, features_from_media, load_media
from .mlops.monitoring import log_prediction
from .mlops.registry import ModelRegistry
from .utils import modality_from_extension, read_json, sha256_bytes

LIMITATIONS = [
    "A statistical model produces a probability, not proof; this result is not legal or forensic evidence.",
    "Reliability is lower on heavily compressed, resized, re-recorded or otherwise transformed media "
    "(see the robustness benchmark).",
    "The detector was trained on a limited set of generators; unseen or newer generation methods may be missed.",
    "Real media can be flagged (false alarms) and manipulated media can pass (misses).",
    "Any evidentiary, journalistic or legal use should be reviewed by a qualified human analyst.",
]


class ForensiXEngine:
    """Loads the production model of every modality lazily and analyses media."""

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self.registry = ModelRegistry(self.cfg)
        self._cache: dict[tuple, dict] = {}

    # ------------------------------------------------------------------ models
    def available_modalities(self) -> list[str]:
        return [m for m in ("image", "audio", "video") if self.registry.production(m)]

    def load(self, modality: str, entry: dict | None = None) -> dict:
        entry = entry or self.registry.production(modality)
        if entry is None:
            raise RuntimeError(f"No production {modality} model. Run the training pipeline first (`make train`).")
        key = (modality, entry["algorithm"], entry["version"])
        if key not in self._cache:
            d = self.registry.model_dir(entry)
            self._cache[key] = {
                "entry": entry,
                "model": joblib.load(d / "model.joblib"),
                "meta": read_json(d / "meta.json"),
                "background": pd.read_csv(d / "background.csv"),
                "reference": read_json(d / "reference_stats.json", {}),
            }
        return self._cache[key]

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ verdict
    def verdict(self, p_fake: float) -> dict:
        lo, hi = self.cfg["decision"]["authentic_below"], self.cfg["decision"]["manipulated_above"]
        if p_fake >= hi:
            label, text = "manipulated", "potentially manipulated"
        elif p_fake < lo:
            label, text = "authentic", "no strong indication of manipulation"
        else:
            label, text = "inconclusive", "inconclusive (the evidence is mixed)"
        conf = abs(p_fake - 0.5) * 2
        strength = "strong" if conf >= 0.7 else "moderate" if conf >= 0.4 else "weak"
        return {"label": label, "text": text, "evidence_strength": strength, "confidence": conf}

    # ------------------------------------------------------------------ scoring
    def score_features(self, feats: dict, modality: str, entry: dict | None = None) -> float:
        bundle = self.load(modality, entry)
        cols = bundle["meta"]["features"]
        x = pd.DataFrame([{c: feats.get(c, 0.0) for c in cols}])[cols].to_numpy(float)
        return float(bundle["model"].predict_proba(x)[0, 1])

    def score_path(self, path, modality: str | None = None) -> dict | None:
        modality = modality or modality_from_extension(path)
        feats = extract_from_path(path, modality, self.cfg)
        if feats is None:
            return None
        return {"modality": modality, "features": feats, "p_fake": self.score_features(feats, modality)}

    # ------------------------------------------------------------------ full analysis
    def analyze_bytes(
        self, data: bytes, filename: str, case_id: str | None = None, explain: bool = True, log: bool = True
    ) -> dict:
        """Analyse an uploaded file. The temporary copy is deleted before returning (privacy by default)."""
        t0 = time.time()
        modality = modality_from_extension(filename)
        if modality is None:
            raise ValueError(f"Unsupported file type: {Path(filename).suffix or filename!r}")
        max_mb = self.cfg["privacy"]["max_upload_mb"]
        if len(data) > max_mb * 1024 * 1024:
            raise ValueError(f"File exceeds the {max_mb} MB limit")
        sha = sha256_bytes(data)
        with tempfile.TemporaryDirectory(prefix="fx_up_") as tmp:
            path = Path(tmp) / f"upload{Path(filename).suffix.lower()}"
            path.write_bytes(data)
            media = load_media(path, modality, self.cfg)
            if media is None:
                raise ValueError("The file could not be read (corrupt, empty, silent or unsupported content).")
            feats = features_from_media(media, modality, self.cfg)
        if feats is None:
            raise ValueError("The media is too short or too degraded to analyse reliably.")
        res = self.analyze_features(
            feats, modality, filename=filename, sha256=sha, case_id=case_id, explain=explain, log=log
        )
        res["media"] = media  # raw in-memory media for the Evidence visualisations; never persisted
        res["elapsed_s"] = round(time.time() - t0, 2)
        return res

    def analyze_features(
        self,
        feats: dict,
        modality: str,
        filename: str | None = None,
        sha256: str | None = None,
        case_id: str | None = None,
        explain: bool = True,
        log: bool = True,
    ) -> dict:
        bundle = self.load(modality)
        entry, meta = bundle["entry"], bundle["meta"]
        cols = meta["features"]
        x = pd.Series({c: feats.get(c, 0.0) for c in cols})
        p = float(bundle["model"].predict_proba(x.to_numpy(float).reshape(1, -1))[0, 1])
        result = {
            "filename": filename,
            "sha256": sha256,
            "modality": modality,
            "probability_manipulated": p,
            "probability_authentic": 1 - p,
            "verdict": self.verdict(p),
            "model": {
                "display": entry["display"],
                "algorithm": entry["algorithm"],
                "version": entry["version"],
                "dataset_version": entry.get("dataset_version"),
                "test_metrics": entry.get("metrics", {}),
            },
            "features": {c: float(x[c]) for c in cols},
            "limitations": LIMITATIONS,
            "case_id": case_id,
        }
        if explain:
            result["explanation"] = explain_sample(
                bundle["model"],
                entry["algorithm"],
                bundle["background"],
                x,
                top_k=self.cfg["shap"]["top_k"],
                reference=bundle["reference"],
            )
            result["model_results"] = self.compare_models(feats, modality)
        if log:
            result["record_id"] = log_prediction(
                self.cfg,
                modality,
                result["features"],
                p,
                result["model"],
                case_id=case_id,
                filename=filename,
                sha256=sha256,
            )
        return result

    def compare_models(self, feats: dict, modality: str) -> list[dict]:
        """Score the latest version of every algorithm (useful context: do the models agree?)."""
        out = []
        for algo in ("logistic_regression", "svm", "random_forest", "xgboost"):
            versions = self.registry.versions(modality, algo)
            versions = [v for v in versions if v["stage"] != "rejected"]
            if not versions:
                continue
            entry = versions[-1]
            try:
                p = self.score_features(feats, modality, entry)
            except Exception:
                continue
            out.append(
                {
                    "model": entry["display"],
                    "algorithm": algo,
                    "probability_manipulated": p,
                    "stage": entry["stage"],
                    "production": entry["stage"] == "production",
                }
            )
        return out
