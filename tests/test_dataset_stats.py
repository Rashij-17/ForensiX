import numpy as np
import pandas as pd

from forensix.data.dataset import infer_label_and_source, load_feature_table, load_metadata
from forensix.stats.analysis import correlation_study, feature_tests


def test_label_inference_from_directories():
    assert infer_label_and_source(("image", "fake", "stable_diffusion", "a.png")) == (1, "stable_diffusion")
    assert infer_label_and_source(("audio", "real", "x.wav")) == (0, "real")
    assert infer_label_and_source(("audio", "bonafide", "mic", "x.wav")) == (0, "mic")
    assert infer_label_and_source(("train", "spoof", "tts_a", "x.wav")) == (1, "tts_a")
    assert infer_label_and_source(("misc", "x.png"))[0] is None


def test_splits_are_disjoint_and_cover_all_usable_samples(project):
    md = load_metadata(project)
    ok = md[(md["status"] == "ok") & (md["split"] != "excluded")]
    assert set(ok["split"]) == {"train", "val", "test"}
    # no duplicate group may straddle splits (no leakage)
    per_group = ok.groupby(["modality", "dup_group"])["split"].nunique()
    assert (per_group == 1).all()
    # no identical file content across splits
    per_md5 = ok.groupby(["modality", "md5"])["split"].nunique()
    assert (per_md5 == 1).all()


def test_each_split_has_both_classes_for_every_modality(project):
    md = load_metadata(project)
    ok = md[(md["status"] == "ok") & (md["split"] != "excluded")]
    for (m, s), g in ok.groupby(["modality", "split"]):
        assert g["label"].nunique() == 2, f"{m}/{s} missing a class"


def test_duplicate_and_silent_files_are_excluded(project):
    md = load_metadata(project)
    assert (md["status_reason"] == "duplicate_of_earlier_sample").any()
    assert (md["status_reason"] == "silent").any()
    assert (md.loc[md["status"] != "ok", "split"] == "excluded").all()


def test_metadata_records_required_columns(project):
    md = load_metadata(project)
    for c in (
        "sample_id",
        "modality",
        "label",
        "source",
        "file_format",
        "width",
        "height",
        "duration_s",
        "preprocessing_version",
    ):
        assert c in md.columns
    assert md["sample_id"].is_unique


def test_hypothesis_tests_detect_real_difference_and_not_noise():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "label": np.r_[np.zeros(n), np.ones(n)],
            "shifted": np.r_[rng.normal(0, 1, n), rng.normal(1.2, 1, n)],
            "noise_a": rng.normal(0, 1, 2 * n),
            "noise_b": rng.normal(0, 1, 2 * n),
        }
    )
    t = feature_tests(df, ["shifted", "noise_a", "noise_b"]).set_index("feature")
    assert t.loc["shifted", "significant"] and t.loc["shifted", "cohens_d"] > 0.8
    assert t.loc["shifted", "ci95_low"] > 0  # CI of mean(fake)-mean(real) excludes zero
    assert not t.loc[["noise_a", "noise_b"], "significant"].any()
    assert 0.5 < t.loc["shifted", "auc_univariate"] <= 1.0


def test_correlation_study_flags_redundant_features():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    df = pd.DataFrame({"a": x, "b": x * 2 + 0.01 * rng.normal(size=300), "c": rng.normal(size=300)})
    out = correlation_study(df, ["a", "b", "c"], threshold=0.9)
    assert any({p["a"], p["b"]} == {"a", "b"} for p in out["redundant_pairs"])


def test_statistics_stage_runs_on_real_pipeline_output(project):
    import json

    for m in ("image", "audio", "video"):
        s = json.loads((project.reports_dir / f"stats_{m}.json").read_text())
        assert s["n_features_tested"] > 20 and s["n_significant_fdr05"] > 0


def test_feature_table_is_aligned_with_metadata(project):
    df = load_feature_table(project, "image")
    assert df["label"].isin([0, 1]).all() and df["split"].isin(["train", "val", "test"]).all()
    assert not df.isna().any().any()


def test_near_duplicate_images_are_grouped_even_with_different_hash_prefix(tmp_path):
    """A slightly re-compressed / brightened copy must land in the same duplicate group (no leakage)."""
    import cv2

    from forensix.data.dataset import _inspect, assign_duplicate_groups
    from forensix.data.synthetic import make_image

    rng = np.random.default_rng(5)
    base = make_image(rng, False)
    other = make_image(rng, True)
    paths = {}
    for name, img, q in (
        ("a.jpg", base, 95),
        ("a_copy.jpg", np.clip(base.astype(int) + 2, 0, 255).astype(np.uint8), 70),
        ("b.png", other, 0),
    ):
        p = tmp_path / name
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q] if q else [])
        paths[name] = p
    from forensix.config import get_config

    rows = [{"abs_path": str(p), "modality": "image"} for p in paths.values()]
    df = pd.DataFrame([{**r, **_inspect(r, get_config())} for r in rows])
    g = assign_duplicate_groups(df)
    assert g.iloc[0] == g.iloc[1] and g.iloc[0] != g.iloc[2]
