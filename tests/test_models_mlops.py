import json

import numpy as np
import pandas as pd
import pytest

from forensix.config import MODEL_NAMES
from forensix.data.dataset import feature_columns, load_feature_table
from forensix.mlops import monitoring as M
from forensix.mlops.registry import ModelRegistry
from forensix.mlops.versioning import list_versions, record_version
from forensix.models.train import comparison_table, evaluate_scores
from forensix.utils import read_json

MODS = ("image", "audio", "video")


def test_evaluate_scores_known_values():
    y = np.array([0, 0, 1, 1])
    out = evaluate_scores(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert out["accuracy"] == 0.5 and out["precision"] == 0.5 and out["recall"] == 0.5 and out["f1"] == 0.5
    assert out["confusion"] == [[1, 1], [1, 1]] and abs(out["roc_auc"] - 0.75) < 1e-9
    assert "roc_curve" in out and "pr_curve" in out


@pytest.mark.parametrize("m", MODS)
def test_all_four_models_trained_per_modality(project, m):
    rep = json.loads((project.reports_dir / f"models_{m}.json").read_text())
    assert set(rep["results"]) == set(MODEL_NAMES)
    for r in rep["results"].values():
        for split in ("train", "val", "test"):
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion"):
                assert k in r[split]


@pytest.mark.parametrize("m", MODS)
def test_winner_is_selected_on_validation_not_test(project, m):
    rep = json.loads((project.reports_dir / f"models_{m}.json").read_text())
    best_val = max(r["val"]["f1"] for r in rep["results"].values())
    assert rep["results"][rep["winner"]]["val"]["f1"] == best_val


@pytest.mark.parametrize("m", MODS)
def test_first_winner_registered_as_the_only_production_model(project, m):
    reg = ModelRegistry(project)
    prod = [v for v in reg.versions(m) if v["stage"] == "production"]
    assert (
        len(prod) == 1
        and prod[0]["algorithm"] == json.loads((project.reports_dir / f"models_{m}.json").read_text())["winner"]
    )
    assert len(reg.versions(m)) == 4 and prod[0]["display"].startswith(m.capitalize())
    assert (reg.model_dir(prod[0]) / "model.joblib").exists() and (reg.model_dir(prod[0]) / "meta.json").exists()


def test_comparison_table_is_complete(project):
    t = comparison_table(project)
    assert len(t) == 12 and t.groupby("modality")["winner"].sum().eq(1).all()


def test_mlflow_runs_record_params_and_metrics(project):
    import mlflow

    from forensix.mlops.tracking import setup_mlflow

    setup_mlflow(project)
    runs = mlflow.search_runs(
        experiment_ids=[e.experiment_id for e in mlflow.search_experiments() if e.name.startswith("forensix-")]
    )
    assert len(runs) >= 12
    for c in (
        "params.modality",
        "params.model",
        "params.feature_set",
        "metrics.test_f1",
        "metrics.test_roc_auc",
        "metrics.test_accuracy",
        "metrics.test_precision",
        "metrics.test_recall",
    ):
        assert c in runs.columns and runs[c].notna().all()


def test_dataset_versioning_is_idempotent_and_detects_change(project):
    from forensix.data.dataset import load_metadata

    md = load_metadata(project)
    n = len(list_versions(project))
    assert n >= 1 and record_version(project, md)["version"] == list_versions(project)[-1]["version"]
    assert len(list_versions(project)) == n  # unchanged -> no new version
    v = record_version(project, md, note="verified labels", n_verified_added=3, extra_fingerprint=["x:3"])
    assert len(list_versions(project)) == n + 1 and v["verified_production_samples_added"] == 3


@pytest.mark.parametrize("m", MODS)
def test_robustness_reports_clean_and_every_transformation(project, m):
    d = pd.read_csv(project.reports_dir / f"robustness_{m}.csv")
    assert d.iloc[0]["transformation"] == "Clean" and len(d) == 7 or (m == "image" and len(d) == 9)
    assert {"f1", "clean_f1", "f1_drop", "n_scored", "n_failed"} <= set(d.columns)
    assert (d["n_scored"] + d["n_failed"] >= 1).all()


def test_robustness_transforms_actually_change_the_media():
    from forensix.robustness import transforms as T

    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    for out in (T.jpeg_image(img, 30), T.resize_image(img, 0.5), T.blur_image(img, 2.0), T.noise_image(img, 10)):
        assert out.shape == img.shape and np.abs(out.astype(int) - img.astype(int)).mean() > 0.5
    y = np.sin(np.linspace(0, 200, 16000)).astype(np.float32)
    noisy = T.bgnoise_audio(y, 10)
    snr = 10 * np.log10(np.mean(y**2) / np.mean((noisy - y) ** 2))
    assert abs(snr - 10) < 1.0  # requested SNR is honoured
    assert len(T.mp3_audio(y, 16000, 32)) == len(y) and len(T.resample_audio(y, 16000, 8000)) == len(y)
    frames = [img] * 12
    assert len(T.fps_reduce_video(frames, 3)) == 4 and T.downscale_video(frames, 0.5)[0].shape == img.shape
    assert len(T.h264_video(frames, 12, 35)) == len(frames)


def test_generator_analysis_has_intervals_and_support_flags(project):
    for m in MODS:
        d = pd.read_csv(project.reports_dir / f"generators_{m}.csv")
        assert len(d) >= 1
        assert ((d["ci95_low"] <= d["detection_rate"]) & (d["detection_rate"] <= d["ci95_high"])).all()
        assert {"low_support", "n_fake_test", "gap_to_best"} <= set(d.columns)
        assert read_json(project.reports_dir / f"generators_{m}.json")["fairness_note"]


def test_global_shap_outputs_exist(project):
    for m in MODS:
        s = read_json(project.reports_dir / "shap" / m / "summary.json")
        assert len(s["top_features"]) >= 5 and (project.reports_dir / "shap" / m / "beeswarm.png").exists()


# ----------------------------------------------------------------------------------- monitoring
def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    edges = np.quantile(base, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    props = np.histogram(base, bins=edges)[0] / len(base)
    assert M.psi(props, rng.normal(0, 1, 2000), edges) < 0.05
    assert M.psi(props, rng.normal(2, 1, 2000), edges) > 0.5


def _log_rows(cfg, modality, feats_df, probs, label=None):
    reg = ModelRegistry(cfg)
    prod = reg.production(modality)
    ids = []
    for (_, r), p in zip(feats_df.iterrows(), probs):
        ids.append(M.log_prediction(cfg, modality, r.to_dict(), float(p), prod))
    if label is not None:
        for i, lab in zip(ids, label):
            M.verify_label(cfg, i, int(lab))
    return ids


def test_drift_not_judged_with_too_few_samples_then_detected_when_shifted(project):
    df = load_feature_table(project, "video")
    feats = feature_columns(df)
    test = df[df["split"] == "test"][feats]
    mon0 = M.compute_monitoring(project, "video")
    assert mon0["baseline"] and not mon0["drift_judged"] and not M.retrain_decision(project, "video")["retrain"]
    shifted = pd.concat([test] * 20, ignore_index=True)
    shifted = shifted * 3.0 + 5.0  # strong covariate shift
    _log_rows(project, "video", shifted.head(60), np.full(60, 0.5))
    mon = M.compute_monitoring(project, "video")
    assert mon["drift_judged"] and mon["drift_share"] > 0.3 and mon["drifted_features"]
    dec = M.retrain_decision(project, "video")
    assert dec["retrain"] and any("drift" in r for r in dec["reasons"])


def test_unshifted_data_does_not_trigger_drift(project):
    df = load_feature_table(project, "audio")
    feats = feature_columns(df)
    train = df[df["split"] == "train"][feats]
    _log_rows(project, "audio", pd.concat([train] * 3, ignore_index=True).head(60), np.full(60, 0.1))
    dec = M.retrain_decision(project, "audio")
    assert dec["monitoring"]["drift_judged"] and not dec["retrain"]


def test_verified_labels_drive_production_metrics_and_degradation_trigger(project):
    df = load_feature_table(project, "image")
    feats = feature_columns(df)
    already_verified = len(M.verified_training_rows(project, "image"))  # other tests may have verified some
    rows = df[df["split"] == "test"][feats]
    rows = pd.concat([rows] * 4, ignore_index=True).head(40)
    labels = np.tile(df[df["split"] == "test"]["label"].to_numpy(), 4)[:40]
    wrong = 1 - labels  # production model "was wrong" on every sample
    _log_rows(project, "image", rows, wrong.astype(float) * 0.9 + 0.05, label=labels)
    perf = M.compute_monitoring(project, "image")["performance"]
    assert perf["judged"] and perf["f1"] < 0.2 and perf["f1_drop"] > 0.05
    dec = M.retrain_decision(project, "image")
    assert dec["retrain"] and any("performance" in r for r in dec["reasons"])
    assert len(M.verified_training_rows(project, "image")) == already_verified + 40


def test_prediction_log_never_stores_media(project):
    p = project.monitoring_dir / "predictions.jsonl"
    assert p.exists()
    first = json.loads(p.read_text().splitlines()[0])
    assert set(first) >= {"id", "ts", "modality", "probability", "features", "model"}
    assert not any(k in first for k in ("media", "bytes", "data", "image", "audio_data"))


def test_retraining_skipped_without_trigger_and_gated_by_improvement(project):
    out = M.run_retraining(project, "audio")
    assert out["action"] == "skipped"
    reg = ModelRegistry(project)
    before = reg.production("audio")
    project.p["monitoring"]["min_improvement"] = 1.0  # impossible to beat -> must be rejected
    try:
        out = M.run_retraining(project, "audio", force=True)
    finally:
        project.p["monitoring"]["min_improvement"] = 0.0
    assert out["action"] == "rejected"
    after = reg.production("audio")
    assert after["version"] == before["version"] and after["algorithm"] == before["algorithm"]
    assert any(v["stage"] == "rejected" for v in reg.versions("audio"))


def test_retraining_promotes_better_candidate_and_archives_old_model(project):
    reg = ModelRegistry(project)
    # make the current production image model deliberately weak by approving the worst candidate
    worst = min(reg.versions("image"), key=lambda v: v["metrics"]["val_f1"])
    reg.set_stage("image", worst["algorithm"], worst["version"], "production", "test: force weak model")
    weak_f1 = worst["metrics"]["test_f1"]
    out = M.run_retraining(project, "image", force=True)
    prod = reg.production("image")
    if out["action"] == "promoted":
        assert out["candidate_test_f1"] > out["production_test_f1"] - 1e-12
        assert any(v["stage"] == "archived" for v in reg.versions("image"))
        assert prod["version"] == out["candidate_version"]
    else:  # weak model was already as good as it gets
        assert out["candidate_test_f1"] <= (out["production_test_f1"] or 0) + 1e-9
    assert len([v for v in reg.versions("image") if v["stage"] == "production"]) == 1
    assert weak_f1 is not None
