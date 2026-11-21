"""Synthetic DEMO dataset generator.

IMPORTANT: this is NOT the DeepSafe dataset. It procedurally creates small image / audio / video files whose
"fake" class carries generator-like artefacts (smoother texture, periodic upsampling peaks, band-limited voice,
temporal flicker) so that the whole ForensiX pipeline can be exercised and tested without downloading anything.
Numbers obtained on this data say nothing about real-world deepfake detection performance.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

# name -> artefact strength (0 weak ... 1 strong). Differences let the generator-wise analysis show variation.
IMAGE_GENERATORS = {"demo_gan": 1.0, "demo_diffusion": 0.55, "demo_upscaler": 0.8}
AUDIO_GENERATORS = {"demo_tts": 1.0, "demo_vocoder": 0.6}
VIDEO_GENERATORS = {"demo_faceswap": 1.0, "demo_t2v": 0.7}


# ---------------------------------------------------------------- images
def _base_scene(rng: np.random.Generator, size: int = 192) -> np.ndarray:
    """Smooth multi-scale 'natural-like' scene with a few shapes (float RGB 0..1)."""
    img = np.zeros((size, size, 3), np.float32)
    for scale, amp in ((6, 0.5), (16, 0.3), (48, 0.15)):
        small = rng.random((scale, scale, 3)).astype(np.float32)
        img += amp * cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    for _ in range(rng.integers(2, 5)):
        c = tuple(int(v) for v in rng.integers(0, size, 2))
        col = tuple(float(v) for v in rng.random(3))
        cv2.circle(img, c, int(rng.integers(10, 40)), col, -1)
    return np.clip(img, 0, 1)


def make_image(rng: np.random.Generator, fake: bool, strength: float = 1.0, size: int = 192) -> np.ndarray:
    img = _base_scene(rng, size)
    if not fake:
        img += rng.normal(0, 0.035, img.shape).astype(np.float32)  # sensor noise
        img += rng.normal(0, 0.012, (size, 1, 3)).astype(np.float32)  # fixed-pattern columns
        out = np.clip(img * 255, 0, 255).astype(np.uint8)
        return out
    img = cv2.GaussianBlur(img, (0, 0), 0.6 + 0.9 * strength)  # over-smooth texture
    img += rng.normal(0, 0.006, img.shape).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    freq = rng.integers(size // 3, size // 2 - 4)
    img += (0.025 * strength * np.sin(2 * np.pi * freq * xx / size))[..., None].astype(np.float32)  # upsampling peak
    img += (0.02 * strength * np.sin(2 * np.pi * (freq - 6) * yy / size))[..., None].astype(np.float32)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- audio
def make_audio(rng: np.random.Generator, fake: bool, strength: float = 1.0, sr: int = 16000, seconds: float = 3.0):
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = rng.uniform(95, 240)
    vib = rng.uniform(3, 6)
    jitter_amt = (0.004 if fake else 0.03 + rng.uniform(0, 0.02)) * (1 - 0.5 * strength * fake)
    drift = np.cumsum(rng.normal(0, jitter_amt, n)) * 0.01
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.01 * np.sin(2 * np.pi * vib * t) + drift)) / sr
    y = np.zeros(n)
    for h in range(1, 14):
        y += (1.0 / h) * np.sin(h * phase + rng.uniform(0, 6.28))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * rng.uniform(1.5, 3.5) * t + rng.uniform(0, 6.28))
    y *= np.clip(env, 0.05, 1)
    if fake:
        spec = np.fft.rfft(y)
        cutoff = int(len(spec) * (0.62 - 0.14 * strength))  # band-limited synthesis
        spec[cutoff:] *= 0.02
        y = np.fft.irfft(spec, n)
        y += rng.normal(0, 0.002, n)
    else:
        y += rng.normal(0, 0.03, n)  # room / mic noise
        y += 0.02 * np.sin(2 * np.pi * 50 * t)
    return (0.8 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32), sr


# ---------------------------------------------------------------- video
def make_video(rng: np.random.Generator, fake: bool, strength: float = 1.0, size: int = 96, n_frames: int = 36):
    base = _base_scene(rng, size)
    pos = rng.uniform(20, size - 20, 2)
    vel = rng.normal(0, 1.6, 2)
    col = tuple(float(v) for v in rng.random(3))
    frames = []
    for _ in range(n_frames):
        pos = pos + vel
        for k in range(2):
            if pos[k] < 12 or pos[k] > size - 12:
                vel[k] *= -1
                pos[k] = np.clip(pos[k], 12, size - 12)
        fr = base.copy()
        cv2.circle(fr, (int(pos[0]), int(pos[1])), 11, col, -1)
        if fake:
            fr = cv2.GaussianBlur(fr, (0, 0), 0.5 + 0.7 * strength)
            fr = fr * (1 + rng.normal(0, 0.05 * strength)) + rng.normal(0, 0.012 * strength)  # flicker
            fr += rng.normal(0, 0.004, fr.shape).astype(np.float32)
            if rng.random() < 0.25 * strength:  # temporal glitch
                fr = np.roll(fr, int(rng.integers(2, 6)), axis=1)
        else:
            fr = fr + rng.normal(0, 0.03, fr.shape).astype(np.float32)
        frames.append(np.clip(fr * 255, 0, 255).astype(np.uint8))
    return frames


def _write_video_h264(frames, path: Path, fps: int = 12) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def generate_demo_dataset(
    raw_dir: str | Path,
    n_image: int = 160,
    n_audio: int = 80,
    n_video: int = 36,
    seed: int = 7,
    duplicates: bool = True,
) -> dict[str, int]:
    """Create ``raw_dir/{image,audio,video}/{real,fake}/<generator>/...``. Half real / half fake per modality."""
    raw = Path(raw_dir)
    rng = np.random.default_rng(seed)
    counts = {"image": 0, "audio": 0, "video": 0}

    def gens(d):
        return list(d.items())

    for i in range(n_image):
        fake = i % 2 == 1
        if fake:
            name, strength = gens(IMAGE_GENERATORS)[(i // 2) % len(IMAGE_GENERATORS)]
            d = raw / "image" / "fake" / name
        else:
            d = raw / "image" / "real" / "demo_camera"
            strength = 0.0
        d.mkdir(parents=True, exist_ok=True)
        img = make_image(rng, fake, strength)
        out = d / f"img_{i:04d}.png" if fake else d / f"img_{i:04d}.jpg"
        cv2.imwrite(
            str(out),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92] if out.suffix == ".jpg" else [],
        )
        counts["image"] += 1
    if duplicates and n_image >= 8:  # plant one exact duplicate so the dedup stage has something to find
        src = next((raw / "image" / "real" / "demo_camera").glob("*.jpg"))
        (raw / "image" / "real" / "demo_camera" / "img_dup_copy.jpg").write_bytes(src.read_bytes())

    for i in range(n_audio):
        fake = i % 2 == 1
        if fake:
            name, strength = gens(AUDIO_GENERATORS)[(i // 2) % len(AUDIO_GENERATORS)]
            d = raw / "audio" / "fake" / name
        else:
            d = raw / "audio" / "real" / "demo_mic"
            strength = 0.0
        d.mkdir(parents=True, exist_ok=True)
        y, sr = make_audio(rng, fake, strength)
        sf.write(str(d / f"aud_{i:04d}.wav"), y, sr)
        counts["audio"] += 1
    # one unusable (silent) file to demonstrate validation
    (raw / "audio" / "real" / "demo_mic").mkdir(parents=True, exist_ok=True)
    sf.write(str(raw / "audio" / "real" / "demo_mic" / "aud_silent.wav"), np.zeros(16000, np.float32), 16000)

    for i in range(n_video):
        fake = i % 2 == 1
        if fake:
            name, strength = gens(VIDEO_GENERATORS)[(i // 2) % len(VIDEO_GENERATORS)]
            d = raw / "video" / "fake" / name
        else:
            d = raw / "video" / "real" / "demo_camera"
            strength = 0.0
        d.mkdir(parents=True, exist_ok=True)
        _write_video_h264(make_video(rng, fake, strength), d / f"vid_{i:04d}.mp4")
        counts["video"] += 1
    (raw / "DEMO_DATA_README.txt").write_text(
        "Synthetic demo data generated by forensix.data.synthetic - NOT the DeepSafe dataset.\n"
        "Delete this folder and place the real DeepSafe files here (see docs/DATASET.md).\n"
    )
    return counts
