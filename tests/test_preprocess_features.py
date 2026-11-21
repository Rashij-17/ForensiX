import numpy as np
import pytest

from forensix.config import get_config
from forensix.features import extract_from_path
from forensix.features.catalog import describe_feature, indicator_group
from forensix.preprocess import audio as pa
from forensix.preprocess import image as pi
from forensix.preprocess import video as pv


def test_corrupt_image_is_rejected(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert pi.load_image(bad) is None
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    assert pi.load_image(empty) is None


def test_image_preprocess_common_resolution_and_range():
    rgb = (np.random.default_rng(0).random((100, 301, 3)) * 255).astype(np.uint8)
    out = pi.preprocess_image(rgb, 64)
    assert out.shape == (64, 64, 3) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_audio_validation_rules():
    sr = 16000
    assert pa.validate_audio(None, sr, 0.5, -55)[1] == "undecodable"
    assert pa.validate_audio(np.zeros(sr, np.float32), sr, 0.5, -55)[1] == "silent"
    assert pa.validate_audio(np.ones(100, np.float32) * 0.1, sr, 0.5, -55)[1] == "too_short"
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    assert pa.validate_audio(tone, sr, 0.5, -55) == (True, "ok")
    nan = tone.copy()
    nan[3] = np.nan
    assert pa.validate_audio(nan, sr, 0.5, -55)[1] == "non_finite"


def test_audio_normalise_and_segment():
    sr = 16000
    y = (np.random.default_rng(0).normal(0, 0.1, sr * 10) + 0.2).astype(np.float32)
    n = pa.normalise_amplitude(y)
    assert abs(float(n.mean())) < 1e-3 and abs(float(np.max(np.abs(n))) - 0.95) < 1e-4
    segs = pa.segment_audio(n, sr, 4.0, 3)
    assert len(segs) == 3 and all(len(s) == 4 * sr for s in segs)
    assert len(pa.segment_audio(n[:sr], sr, 4.0, 3)) == 1  # short clip kept as-is


def test_corrupt_video_is_rejected(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00" * 2048)
    assert pv.probe_video(bad) is None


def test_video_sampling_fixed_interval():
    frames = [np.full((40, 60, 3), i, np.uint8) for i in range(20)]
    arr = pv.sample_frames(frames, interval=4, max_frames=10, size=32)
    assert arr.shape == (5, 32, 32, 3) and arr.dtype == np.float32


@pytest.mark.parametrize(
    "name,modality", [("fake.png", "image"), ("real.jpg", "image"), ("fake.wav", "audio"), ("fake.mp4", "video")]
)
def test_features_finite_and_deterministic(sample_files, name, modality):
    cfg = get_config()
    a = extract_from_path(sample_files / name, modality, cfg)
    b = extract_from_path(sample_files / name, modality, cfg)
    assert a is not None and len(a) > 25
    assert all(np.isfinite(v) for v in a.values())
    assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-9 for k in a)


def test_feature_catalog_covers_all_features(sample_files):
    cfg = get_config()
    for name, modality in (("fake.png", "image"), ("fake.wav", "audio"), ("fake.mp4", "video")):
        for f in extract_from_path(sample_files / name, modality, cfg):
            group, desc = describe_feature(f)
            assert group != "Other", f"feature {f} has no catalog entry"
            assert indicator_group(f) == group and desc


def test_silent_audio_gives_no_features(tmp_path):
    import soundfile as sf

    p = tmp_path / "s.wav"
    sf.write(str(p), np.zeros(16000, np.float32), 16000)
    assert extract_from_path(p, "audio", get_config()) is None
