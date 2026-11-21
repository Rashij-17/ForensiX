"""Video preprocessing: corruption check, fixed-interval frame sampling, resizing, normalisation."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def probe_video(path) -> dict | None:
    """Return basic stream info or None if the file cannot be opened / has no frames."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if not ok or n <= 0 or w <= 0 or h <= 0:
            return None
        return {
            "frames": n,
            "fps": fps if fps > 0 else 25.0,
            "width": w,
            "height": h,
            "duration": n / fps if fps > 0 else 0.0,
        }
    finally:
        cap.release()


def read_raw_frames(path, max_raw_frames: int = 120, max_side: int = 320) -> tuple[list[np.ndarray], float] | None:
    """Read up to ``max_raw_frames`` RGB uint8 frames (longest side <= ``max_side``). Returns (frames, fps)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_raw_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1.0:
                bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return (frames, fps) if frames else None


def sample_frames(frames: list[np.ndarray], interval: int, max_frames: int, size: int) -> np.ndarray:
    """Fixed-interval sampling, resize to ``size`` x ``size`` and normalise to float32 [0, 1]; shape (T, H, W, 3)."""
    picked = frames[:: max(1, interval)][:max_frames]
    out = [cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in picked]
    return (np.stack(out).astype(np.float32) / 255.0) if out else np.zeros((0, size, size, 3), np.float32)


def write_video(frames: list[np.ndarray], fps: float, path) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def ffmpeg_h264_roundtrip(frames: list[np.ndarray], fps: float, crf: int) -> list[np.ndarray]:
    """Encode with libx264 at ``crf`` and decode again; falls back to per-frame JPEG if ffmpeg is unavailable."""
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2  # libx264 needs even dimensions
    if shutil.which("ffmpeg") is None:
        return _jpeg_fallback(frames, quality=max(5, 100 - crf * 2))
    tmpdir = tempfile.mkdtemp(prefix="fx_vid_")
    out_path = os.path.join(tmpdir, "rt.mp4")
    try:
        raw = np.stack([f[:h2, :w2] for f in frames]).astype(np.uint8).tobytes()
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w2}x{h2}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        subprocess.run(cmd, input=raw, check=True, timeout=180)
        res = read_raw_frames(out_path, max_raw_frames=len(frames), max_side=10_000)
        return res[0] if res else _jpeg_fallback(frames, quality=max(5, 100 - crf * 2))
    except Exception:
        return _jpeg_fallback(frames, quality=max(5, 100 - crf * 2))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _jpeg_fallback(frames: list[np.ndarray], quality: int) -> list[np.ndarray]:
    out = []
    for f in frames:
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        out.append(cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else f)
    return out
