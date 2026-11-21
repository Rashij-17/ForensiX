"""Model registry.

A small JSON registry (``artifacts/registry.json``) is the source of truth for named, versioned models and for
which version is approved for production. Every registration is mirrored into the MLflow Model Registry when
MLflow is available (best effort: the JSON registry never depends on it).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from ..config import MODEL_LABELS
from ..utils import read_json, write_json

STAGES = ("staging", "production", "archived", "rejected")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def display_name(modality: str, algorithm: str, version: int) -> str:
    return f"{modality.capitalize()} {MODEL_LABELS.get(algorithm, algorithm)} v{version}"


class ModelRegistry:
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = cfg.artifacts_dir / "registry.json"

    # ---- persistence --------------------------------------------------------
    def _load(self) -> dict:
        return read_json(self.path, {"models": []})

    def _save(self, data: dict) -> None:
        write_json(self.path, data)

    # ---- queries ---------------------------------------------------------------
    def all(self) -> list[dict]:
        return self._load()["models"]

    def versions(self, modality: str, algorithm: str | None = None) -> list[dict]:
        return [
            m for m in self.all() if m["modality"] == modality and (algorithm is None or m["algorithm"] == algorithm)
        ]

    def production(self, modality: str) -> dict | None:
        prod = [m for m in self.versions(modality) if m["stage"] == "production"]
        return prod[-1] if prod else None

    def get(self, modality: str, algorithm: str, version: int) -> dict | None:
        for m in self.versions(modality, algorithm):
            if m["version"] == version:
                return m
        return None

    def next_version(self, modality: str, algorithm: str) -> int:
        return max([m["version"] for m in self.versions(modality, algorithm)] or [0]) + 1

    def model_dir(self, entry: dict) -> Path:
        return self.cfg.root / entry["path"]

    # ---- mutations --------------------------------------------------------------
    def register(
        self,
        modality: str,
        algorithm: str,
        metrics: dict,
        rel_path: str,
        version: int,
        dataset_version: str,
        feature_version: str,
        run_id: str | None = None,
        stage: str = "staging",
        notes: str = "",
        cycle: str = "",
    ) -> dict:
        data = self._load()
        entry = {
            "name": f"{modality.capitalize()} {MODEL_LABELS.get(algorithm, algorithm)}",
            "display": display_name(modality, algorithm, version),
            "modality": modality,
            "algorithm": algorithm,
            "version": version,
            "stage": stage,
            "cycle": cycle,
            "metrics": metrics,
            "path": rel_path,
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "mlflow_run_id": run_id,
            "created": _now(),
            "notes": notes,
            "approved_at": _now() if stage == "production" else None,
        }
        data["models"].append(entry)
        self._save(data)
        self._mirror_to_mlflow(entry)
        return entry

    def set_stage(self, modality: str, algorithm: str, version: int, stage: str, note: str = "") -> dict:
        """Change a version's stage. Promoting to production archives the previous production version."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        data = self._load()
        target = None
        for m in data["models"]:
            if m["modality"] == modality and m["algorithm"] == algorithm and m["version"] == version:
                target = m
        if target is None:
            raise KeyError(f"{modality}/{algorithm} v{version} is not registered")
        if stage == "production":
            for m in data["models"]:
                if m["modality"] == modality and m["stage"] == "production" and m is not target:
                    m["stage"] = "archived"
                    m["notes"] = (m.get("notes", "") + f" | superseded {_now()}").strip(" |")
            target["approved_at"] = _now()
        target["stage"] = stage
        if note:
            target["notes"] = (target.get("notes", "") + f" | {note}").strip(" |")
        self._save(data)
        self._mirror_stage(target)
        return target

    # ---- MLflow mirror (best effort) ----------------------------------------------
    def _mirror_to_mlflow(self, entry: dict) -> None:
        try:
            from mlflow.tracking import MlflowClient

            from .tracking import setup_mlflow

            setup_mlflow(self.cfg)
            client = MlflowClient()
            name = entry["name"].replace(" ", "-")
            try:
                client.create_registered_model(name)
            except Exception:
                pass
            client.create_model_version(
                name=name,
                source=str(self.cfg.root / entry["path"]),
                run_id=entry.get("mlflow_run_id"),
                tags={
                    "modality": entry["modality"],
                    "algorithm": entry["algorithm"],
                    "stage": entry["stage"],
                    "forensix_version": str(entry["version"]),
                },
            )
        except Exception:
            pass

    def _mirror_stage(self, entry: dict) -> None:
        try:
            from mlflow.tracking import MlflowClient

            from .tracking import setup_mlflow

            setup_mlflow(self.cfg)
            client = MlflowClient()
            name = entry["name"].replace(" ", "-")
            for mv in client.search_model_versions(f"name='{name}'"):
                if mv.tags.get("forensix_version") == str(entry["version"]):
                    client.set_model_version_tag(name, mv.version, "stage", entry["stage"])
        except Exception:
            pass
