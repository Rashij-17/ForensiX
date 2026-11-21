"""Model monitoring: input drift (PSI), prediction / confidence distributions, production metrics from verified labels.

Every analysis served by the app is appended to ``artifacts/monitoring/predictions.jsonl`` (features, probability,
timestamp, model version; NEVER the uploaded media). A human can later attach a verified label to a logged
prediction; those labels feed production F1 / precision / recall and, on retraining, the training set.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from ..utils import json_safe, read_json, write_json

_LOCK = threading.Lock()


def _log_path(cfg):
    return cfg.monitoring_dir / "predictions.jsonl"


def _baseline_path(cfg, modality: str):
    return cfg.monitoring_dir / f"baseline_{modality}.json"


# ---------------------------------------------------------------- baseline
def _full_edges(interior) -> np.ndarray:
    return np.concatenate([[-np.inf], np.asarray(interior, float), [np.inf]])


def save_baseline(cfg, modality: str, train_features: pd.DataFrame, val_probs: np.ndarray, test_f1: float) -> None:
    """Reference distributions captured when a model is approved for production."""
    edges, props = {}, {}
    for c in train_features.columns:
        x = train_features[c].to_numpy(float)
        qs = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        if len(qs) < 3:
            continue
        # Only the finite interior edges are stored: JSON cannot represent +/-inf (it would become null), so the
        # open-ended outer bins are re-added by ``_full_edges`` when the baseline is read back.
        edges[c] = qs[1:-1].tolist()
        props[c] = (np.histogram(x, bins=_full_edges(edges[c]))[0] / len(x)).tolist()
    write_json(
        _baseline_path(cfg, modality),
        {
            "modality": modality,
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n": int(len(train_features)),
            "edges": edges,
            "props": props,
            "test_f1": float(test_f1),
            "prob_hist": np.histogram(val_probs, bins=10, range=(0, 1))[0].tolist(),
            "prob_mean": float(np.mean(val_probs)),
            "conf_mean": float(np.mean(np.abs(np.asarray(val_probs) - 0.5) * 2)),
            "feature_mean": {c: float(train_features[c].mean()) for c in train_features.columns},
        },
    )


# ---------------------------------------------------------------- logging
def log_prediction(
    cfg,
    modality: str,
    features: dict,
    probability: float,
    model: dict,
    case_id: str | None = None,
    filename: str | None = None,
    sha256: str | None = None,
) -> str:
    """Append one prediction record and return its id. The uploaded media itself is never stored here."""
    rec_id = uuid.uuid4().hex[:12]
    rec = {
        "id": rec_id,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "modality": modality,
        "probability": float(probability),
        "model": model.get("display"),
        "algorithm": model.get("algorithm"),
        "version": model.get("version"),
        "case_id": case_id,
        "filename": filename,
        "sha256": sha256,
        "features": features,
        "verified_label": None,
    }
    cfg.monitoring_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(_log_path(cfg), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(rec)) + "\n")
    return rec_id


def read_log(cfg) -> pd.DataFrame:
    p = _log_path(cfg)
    if not p.exists():
        return pd.DataFrame(columns=["id", "ts", "modality", "probability", "model", "verified_label", "features"])
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def verify_label(cfg, record_id: str, label: int) -> bool:
    """Attach a human-verified label (0 = authentic, 1 = manipulated) to a logged prediction."""
    p = _log_path(cfg)
    if not p.exists():
        return False
    found = False
    with _LOCK:
        lines = p.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] == record_id:
                rec["verified_label"] = int(label)
                rec["verified_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                found = True
            out.append(json.dumps(json_safe(rec)))
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return found


# ---------------------------------------------------------------- drift
def psi(expected_props: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
    exp = np.clip(np.asarray(expected_props, float), 1e-4, None)
    act = np.histogram(actual, bins=edges)[0].astype(float)
    act = np.clip(act / max(act.sum(), 1), 1e-4, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def compute_monitoring(cfg, modality: str) -> dict:
    """Drift, distribution and (if verified labels exist) production-performance summary for one modality."""
    mc = cfg["monitoring"]
    base = read_json(_baseline_path(cfg, modality))
    log = read_log(cfg)
    log = log[log["modality"] == modality].tail(mc["window"]) if len(log) else log
    out = {
        "modality": modality,
        "n_logged": int(len(log)),
        "baseline": bool(base),
        "min_samples": mc["min_samples"],
        "drift_judged": False,
        "drifted_features": [],
        "feature_psi": {},
        "drift_share": 0.0,
        "psi_threshold": mc["psi_threshold"],
        "prediction_stats": {},
        "performance": {},
    }
    if len(log):
        probs = log["probability"].to_numpy(float)
        out["prediction_stats"] = {
            "prob_mean": float(probs.mean()),
            "share_flagged": float((probs >= 0.5).mean()),
            "conf_mean": float((np.abs(probs - 0.5) * 2).mean()),
            "prob_hist": np.histogram(probs, bins=10, range=(0, 1))[0].tolist(),
            "baseline_prob_hist": base["prob_hist"] if base else None,
            "baseline_conf_mean": base["conf_mean"] if base else None,
            "baseline_prob_mean": base["prob_mean"] if base else None,
        }
    if base and len(log) >= mc["min_samples"]:
        feats = pd.DataFrame(list(log["features"]))
        for c, edges in base["edges"].items():
            if c in feats.columns:
                out["feature_psi"][c] = psi(np.array(base["props"][c]), feats[c].to_numpy(float), _full_edges(edges))
        drifted = [c for c, v in out["feature_psi"].items() if v > mc["psi_threshold"]]
        out["drifted_features"] = sorted(drifted, key=lambda c: -out["feature_psi"][c])
        out["drift_share"] = len(drifted) / max(len(out["feature_psi"]), 1)
        out["drift_judged"] = True
    if len(log) and "verified_label" in log:
        lab = log[log["verified_label"].notna()]
        if len(lab) >= 1:
            y, p = lab["verified_label"].astype(int).to_numpy(), (lab["probability"].to_numpy(float) >= 0.5).astype(int)
            judged = len(lab) >= mc["min_labeled"]
            out["performance"] = {
                "n_labeled": int(len(lab)),
                "judged": judged,
                "f1": float(f1_score(y, p, zero_division=0)),
                "precision": float(precision_score(y, p, zero_division=0)),
                "recall": float(recall_score(y, p, zero_division=0)),
                "baseline_f1": base["test_f1"] if base else None,
            }
            if judged and base:
                out["performance"]["f1_drop"] = float(base["test_f1"] - out["performance"]["f1"])
    return out


def retrain_decision(cfg, modality: str) -> dict:
    """Decide whether retraining is warranted. Never claims drift without enough data (see ``min_samples``)."""
    mc = cfg["monitoring"]
    mon = compute_monitoring(cfg, modality)
    reasons = []
    if mon["drift_judged"] and mon["drift_share"] >= mc["drift_share_trigger"]:
        reasons.append(
            f"data drift: {len(mon['drifted_features'])} features ({mon['drift_share']:.0%}) exceed PSI "
            f"{mc['psi_threshold']}"
        )
    perf = mon["performance"]
    if perf.get("judged") and perf.get("f1_drop", 0) > mc["f1_drop_trigger"]:
        reasons.append(
            f"performance degradation: production F1 {perf['f1']:.3f} is {perf['f1_drop']:.3f} below baseline"
        )
    return {"modality": modality, "retrain": bool(reasons), "reasons": reasons, "monitoring": mon}


def verified_training_rows(cfg, modality: str) -> pd.DataFrame:
    """Verified production samples as training rows (features + label)."""
    log = read_log(cfg)
    if log.empty:
        return pd.DataFrame()
    lab = log[(log["modality"] == modality) & log["verified_label"].notna()]
    if lab.empty:
        return pd.DataFrame()
    feats = pd.DataFrame(list(lab["features"]))
    feats["label"] = lab["verified_label"].astype(int).to_numpy()
    return feats


def run_retraining(cfg, modality: str, force: bool = False) -> dict:
    """Retraining cycle: train candidates, promote ONLY if the best candidate beats the current production model.

    Fair comparison: both the production model and the candidates are scored on the same untouched TEST split.
    """
    from ..data.dataset import feature_columns, load_feature_table, load_metadata
    from ..models.train import evaluate_scores, train_modality
    from ..mlops.registry import ModelRegistry
    from ..mlops.versioning import record_version

    decision = retrain_decision(cfg, modality)
    if not decision["retrain"] and not force:
        return {
            "action": "skipped",
            "modality": modality,
            "reason": "no significant drift or performance degradation",
            "decision": decision,
        }
    reg = ModelRegistry(cfg)
    prod = reg.production(modality)
    extra = verified_training_rows(cfg, modality)
    df = load_feature_table(cfg, modality)
    feats = feature_columns(df)
    if len(extra):
        extra = extra.reindex(columns=feats + ["label"]).fillna(0.0)
    version = record_version(
        cfg,
        load_metadata(cfg),
        note=f"retraining cycle for {modality}",
        n_verified_added=len(extra),
        extra_fingerprint=[f"{modality}:{len(extra)}"],
    )
    report = train_modality(
        cfg,
        modality,
        extra_train=extra if len(extra) else None,
        promote_first=False,
        note="retraining candidate",
        dataset_version=version["version"],
    )
    winner = report["winner"]
    cand_f1 = report["results"][winner]["test"]["f1"]
    prod_f1 = None
    if prod is not None:
        import joblib

        pdir = reg.model_dir(prod)
        meta = read_json(pdir / "meta.json")
        te = df[df["split"] == "test"]
        model = joblib.load(pdir / "model.joblib")
        pf = meta["features"]
        prod_f1 = evaluate_scores(
            te["label"].to_numpy(int), model.predict_proba(te[pf].to_numpy(float))[:, 1], curves=False
        )["f1"]
    improved = prod_f1 is None or cand_f1 > prod_f1 + mc_min(cfg)
    cand_version = report["versions"][winner]
    if improved:
        reg.set_stage(
            modality,
            winner,
            cand_version,
            "production",
            f"promoted by retraining: test F1 {cand_f1:.3f} vs {prod_f1 if prod_f1 is None else round(prod_f1, 3)}",
        )
        tr = df[df["split"] == "train"]
        val = df[df["split"] == "val"]
        import joblib

        m = joblib.load(reg.model_dir(reg.get(modality, winner, cand_version)) / "model.joblib")
        save_baseline(cfg, modality, tr[feats], m.predict_proba(val[feats].to_numpy(float))[:, 1], cand_f1)
    else:
        for name in report["versions"]:
            reg.set_stage(
                modality,
                name,
                report["versions"][name],
                "rejected",
                f"did not beat production (candidate F1 {cand_f1:.3f} <= {prod_f1:.3f})",
            )
    return {
        "action": "promoted" if improved else "rejected",
        "modality": modality,
        "winner": winner,
        "candidate_version": cand_version,
        "candidate_test_f1": cand_f1,
        "production_test_f1": prod_f1,
        "verified_samples_used": int(len(extra)),
        "dataset_version": version["version"],
        "decision": decision,
    }


def mc_min(cfg) -> float:
    return float(cfg["monitoring"].get("min_improvement", 0.0))
