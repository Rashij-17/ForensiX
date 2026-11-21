"""Shared fixtures. The ``project`` fixture builds a small synthetic project once per test session."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """A throw-away project root with demo data, features, trained models and all reports."""
    from forensix.config import get_config
    from forensix.data.synthetic import generate_demo_dataset
    from forensix.pipeline import main as run

    root = tmp_path_factory.mktemp("fx_project")
    (root / "config").mkdir()
    shutil.copy(REPO / "config" / "params.yaml", root / "config" / "params.yaml")
    cfg = get_config(root)
    generate_demo_dataset(cfg.raw_dir, n_image=64, n_audio=40, n_video=24)
    run(
        [
            "ingest",
            "features",
            "stats",
            "train",
            "explain",
            "robustness",
            "generators",
            "--root",
            str(root),
            "--jobs",
            "2",
        ]
    )
    return get_config(root)


@pytest.fixture(scope="session")
def sample_files(tmp_path_factory):
    import cv2
    import numpy as np
    import soundfile as sf

    from forensix.data.synthetic import make_audio, make_image, make_video

    d = tmp_path_factory.mktemp("samples")
    rng = np.random.default_rng(1234)
    cv2.imwrite(str(d / "fake.png"), cv2.cvtColor(make_image(rng, True), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(d / "real.jpg"), cv2.cvtColor(make_image(rng, False), cv2.COLOR_RGB2BGR))
    y, sr = make_audio(rng, True)
    sf.write(str(d / "fake.wav"), y, sr)
    frames = make_video(rng, True)
    w = cv2.VideoWriter(
        str(d / "fake.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (frames[0].shape[1], frames[0].shape[0])
    )
    for f in frames:
        w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    w.release()
    return d
