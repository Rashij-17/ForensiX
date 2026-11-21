"""Service layer shared by the FastAPI backend and the in-process (standalone) Streamlit mode.

Every public method takes plain Python values and returns JSON-serialisable data, so the same functions sit behind
both the HTTP API and the local backend. Uploaded media is held in memory / a temp dir only for the call.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import lab as L
from . import visuals as V
from .cases import CaseStore
from .config import get_config
from .features import load_media
from .inference import ForensiXEngine
from .mlops import monitoring as M
from .mlops.registry import ModelRegistry
from .mlops.versioning import list_versions
from .models.train import comparison_table
from .report import build_report, case_report
from .utils import fig_to_b64, json_safe, modality_from_extension, read_json


def _load_bytes(data: bytes, filename: str, cfg):
    modality = modality_from_extension(filename)
    if modality is None:
        raise ValueError(f"Unsupported file type: {Path(filename).suffix or filename!r}")
    with tempfile.TemporaryDirectory(prefix="fx_ld_") as tmp:
        p = Path(tmp) / f"f{Path(filename).suffix.lower()}"
        p.write_bytes(data)
        media = load_media(p, modality, cfg)
    if media is None:
        raise ValueError(f"{filename}: the file could not be read (corrupt, empty, silent or unsupported)")
    return modality, media


def _png_b64(arr: np.ndarray) -> str:
    import cv2

    ok, enc = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(enc.tobytes()).decode("ascii")


class ForensiXService:
    def __init__(self, root=None):
        self.cfg = get_config(root)
        self.engine = ForensiXEngine(self.cfg)
        self.cases = CaseStore(self.cfg)
        self.registry = ModelRegistry(self.cfg)

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        prod = {m: self.registry.production(m) for m in ("image", "audio", "video")}
        return json_safe(
            {
                "ready": any(prod.values()),
                "modalities": self.engine.available_modalities(),
                "demo_data": (self.cfg.raw_dir / "DEMO_DATA_README.txt").exists(),
                "production": {m: (p["display"] if p else None) for m, p in prod.items()},
                "dataset_versions": list_versions(self.cfg),
            }
        )

    def reload(self) -> None:
        self.engine.clear_cache()

    # ------------------------------------------------------------------ analyse
    def analyze(
        self, data: bytes, filename: str, case_id: str | None = None, explain: bool = True, note: str = ""
    ) -> dict:
        res = self.engine.analyze_bytes(data, filename, case_id=case_id, explain=explain)
        res.pop("media", None)
        if case_id:
            self.cases.add_analysis(case_id, res, note)
        return json_safe(res)

    def evidence(self, data: bytes, filename: str) -> dict[str, str]:
        """Evidence visualisations as base64 PNGs: FFT/DCT (image), spectrogram/MFCC (audio), temporal (video)."""
        modality, media = _load_bytes(data, filename, self.cfg)
        figs: dict[str, plt.Figure]
        if modality == "image":
            figs = V.image_evidence_figs(media, self.cfg.get("image", "size"))
        elif modality == "audio":
            y = media[: self.cfg["audio"]["sr"] * 8]
            figs = V.audio_evidence_figs(y, self.cfg["audio"]["sr"])
        else:
            figs = V.video_evidence_figs(media[0], self.cfg)
        return {k: fig_to_b64(f) for k, f in figs.items()}

    def shap_figure(self, result: dict) -> str:
        return fig_to_b64(V.shap_bar(result["explanation"]["contributions"]))

    # ------------------------------------------------------------------ compare
    def compare(self, data_a: bytes, name_a: str, data_b: bytes, name_b: str) -> dict:
        ma, a = _load_bytes(data_a, name_a, self.cfg)
        mb, b = _load_bytes(data_b, name_b, self.cfg)
        if ma != mb:
            raise ValueError(f"Both files must be the same modality (got {ma} and {mb})")
        scores = L.compare_scores(self.engine, ma, a, b)
        if ma == "image":
            fig, metrics = V.difference_heatmap(a, b, self.cfg.get("image", "size"))
        elif ma == "audio":
            fig, metrics = V.waveform_compare(a, b, self.cfg["audio"]["sr"])
        else:
            fig, metrics = V.video_compare(a[0], b[0], self.cfg["video"]["size"])
        return json_safe({"modality": ma, "scores": scores, "similarity": metrics, "figure": fig_to_b64(fig)})

    # ------------------------------------------------------------------ forensic lab
    def lab_options(self, modality: str) -> dict:
        return L.lab_options(modality)

    def lab(self, data: bytes, filename: str, kind: str, strengths: list[float] | None = None) -> dict:
        modality, media = _load_bytes(data, filename, self.cfg)
        opts = L.lab_options(modality)
        if kind not in opts:
            raise ValueError(f"unknown degradation {kind!r} for {modality}; options: {list(opts)}")
        o = opts[kind]
        strengths = strengths or [float(x) for x in np.linspace(o["max"], o["min"], 6)]
        curve = L.stress_curve(self.engine, modality, media, kind, strengths)
        out = {"modality": modality, "kind": kind, "label": o["label"], "curve": curve}
        if modality == "image":
            mid = strengths[len(strengths) // 2]
            out["preview"] = {
                "strength": mid,
                "png_b64": _png_b64(L.apply_transform("image", media, kind, mid, self.cfg)),
            }
        return json_safe(out)

    # ------------------------------------------------------------------ reports
    def report(self, data: bytes, filename: str, case_id: str | None = None, analyst: str = "") -> dict:
        res = self.engine.analyze_bytes(data, filename, case_id=case_id, explain=True, log=False)
        media = res.pop("media")
        figs = {}
        try:
            if res["modality"] == "image":
                figs.update(
                    {
                        "Frequency-domain evidence": V.image_evidence_figs(media, self.cfg.get("image", "size"))[
                            "frequency"
                        ]
                    }
                )
            elif res["modality"] == "audio":
                figs.update(
                    {
                        "Spectral evidence": V.audio_evidence_figs(
                            media[: self.cfg["audio"]["sr"] * 8], self.cfg["audio"]["sr"]
                        )["spectral"]
                    }
                )
            else:
                figs.update({"Temporal evidence": V.video_evidence_figs(media[0], self.cfg)["temporal"]})
            figs["SHAP explanation"] = V.shap_bar(res["explanation"]["contributions"])
        except Exception:
            pass
        case = self.cases.get(case_id) if case_id else None
        rep = build_report(res, self.cfg, case=case, analyst=analyst, figures=figs)
        return {"markdown": rep["markdown"], "html": rep["html"], "summary": json_safe(rep["json"])}

    # ------------------------------------------------------------------ cases
    def create_case(self, name: str, description: str = "", analyst: str = "") -> dict:
        return json_safe(self.cases.create(name, description, analyst))

    def list_cases(self) -> list[dict]:
        return json_safe(self.cases.list())

    def get_case(self, case_id: str) -> dict:
        return json_safe(self.cases.get(case_id))

    def set_case_status(self, case_id: str, status: str) -> dict:
        return json_safe(self.cases.set_status(case_id, status))

    def add_case_note(self, case_id: str, text: str) -> dict:
        return json_safe(self.cases.add_note(case_id, text))

    def delete_case(self, case_id: str) -> None:
        self.cases.delete(case_id)

    def case_report(self, case_id: str) -> dict:
        return case_report(self.cases.get(case_id), self.cfg)

    # ------------------------------------------------------------------ benchmarks
    def benchmarks(self) -> dict:
        rep = self.cfg.reports_dir
        out: dict = {"model_comparison": [], "robustness": [], "generators": [], "statistics": {}, "shap": {}}
        t = comparison_table(self.cfg)
        out["model_comparison"] = t.to_dict("records") if not t.empty else []
        for m in ("image", "audio", "video"):
            for key, name in (("robustness", f"robustness_{m}.csv"), ("generators", f"generators_{m}.csv")):
                if (rep / name).exists():
                    out[key] += pd.read_csv(rep / name).to_dict("records")
            s = read_json(rep / f"stats_{m}.json")
            if s:
                out["statistics"][m] = s
            g = read_json(rep / "shap" / m / "summary.json")
            if g:
                out["shap"][m] = g
        return json_safe(out)

    def feature_stats(self, modality: str) -> list[dict]:
        p = self.cfg.reports_dir / f"stats_{modality}.csv"
        return json_safe(pd.read_csv(p).to_dict("records")) if p.exists() else []

    # ------------------------------------------------------------------ MLOps
    def registry_list(self) -> list[dict]:
        return json_safe(self.registry.all())

    def promote(self, modality: str, algorithm: str, version: int) -> dict:
        e = self.registry.set_stage(modality, algorithm, version, "production", "manually approved")
        self.reload()
        return json_safe(e)

    def monitoring(self, modality: str) -> dict:
        return json_safe(M.compute_monitoring(self.cfg, modality))

    def prediction_log(self, limit: int = 200) -> list[dict]:
        df = M.read_log(self.cfg)
        if df.empty:
            return []
        return json_safe(df.drop(columns=["features"], errors="ignore").tail(limit).to_dict("records"))

    def feedback(self, record_id: str, label: int) -> dict:
        if label not in (0, 1):
            raise ValueError("label must be 0 (authentic) or 1 (manipulated)")
        return {"updated": M.verify_label(self.cfg, record_id, label)}

    def retrain_check(self, modality: str) -> dict:
        d = M.retrain_decision(self.cfg, modality)
        return json_safe(d)

    def retrain(self, modality: str, force: bool = False) -> dict:
        out = M.run_retraining(self.cfg, modality, force=force)
        self.reload()
        return json_safe(out)

    def dataset_versions(self) -> list[dict]:
        return json_safe(list_versions(self.cfg))

    def experiments(self, limit: int = 200) -> list[dict]:
        """MLflow runs (model, modality, hyper-parameters, metrics) flattened for display."""
        try:
            import mlflow

            from .mlops.tracking import setup_mlflow

            setup_mlflow(self.cfg)
            exps = [e.experiment_id for e in mlflow.search_experiments() if e.name.startswith("forensix-")]
            if not exps:
                return []
            runs = mlflow.search_runs(experiment_ids=exps, max_results=limit, order_by=["attributes.start_time DESC"])
        except Exception:
            return []
        keep = [
            "run_id",
            "start_time",
            "tags.mlflow.runName",
            "params.modality",
            "params.model",
            "params.feature_set",
            "params.dataset_version",
            "params.n_train",
            "metrics.val_accuracy",
            "metrics.val_precision",
            "metrics.val_recall",
            "metrics.val_f1",
            "metrics.val_roc_auc",
            "metrics.test_accuracy",
            "metrics.test_precision",
            "metrics.test_recall",
            "metrics.test_f1",
            "metrics.test_roc_auc",
            "metrics.cv_f1",
        ]
        runs = runs[[c for c in keep if c in runs.columns]].copy()
        runs.columns = [c.split(".", 1)[-1] for c in runs.columns]
        runs["start_time"] = runs["start_time"].astype(str)
        hp = [c for c in runs.columns if c.startswith("hp_")]
        return json_safe(runs.drop(columns=hp, errors="ignore").to_dict("records"))
