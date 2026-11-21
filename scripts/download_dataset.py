#!/usr/bin/env python3
"""Download the DeepSafe benchmark from HuggingFace and arrange it under data/raw/.

    python scripts/download_dataset.py                       # download + arrange
    python scripts/download_dataset.py --src /path/to/local  # arrange an already-downloaded copy
    python scripts/download_dataset.py --dry-run             # only report what would be done

The target layout (which `forensix.data.scan_raw` understands) is

    data/raw/<image|audio|video>/<real|fake>/<source or generator>/<file>

Two source layouts are handled:
  1. media files in folders whose names contain real / fake (copied or linked as they are, regrouped by modality);
  2. parquet / arrow files holding the media bytes plus a label (and optionally a source) column, which are
     exported to individual files.

IMPORTANT: the exact structure of the HuggingFace repository could not be inspected when this script was written, so
the column / folder detection is heuristic. ALWAYS read the summary printed at the end. If labels cannot be inferred,
create data/raw/metadata.csv (columns: path,label[,source]) by hand; see docs/DATASET.md.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from forensix.data.dataset import FAKE_TOKENS, REAL_TOKENS, _norm_label, infer_label_and_source  # noqa: E402
from forensix.utils import modality_from_extension  # noqa: E402

REPO_ID = "siddharthksah/DeepSafe-benchmark"
LABEL_HINTS = ("label", "class", "target", "is_fake", "fake", "real", "authentic")
SOURCE_HINTS = ("source", "generator", "method", "model", "attack", "technique", "origin")


def download(dest: Path, repo_id: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub first:  pip install huggingface_hub")
    print(f"downloading {repo_id} -> {dest} (this can be large)")
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dest)))


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))[:60] or "unknown"


def _sniff_ext(b: bytes) -> str | None:
    sig = {
        b"\xff\xd8\xff": ".jpg",
        b"\x89PNG": ".png",
        b"RIFF": ".wav",
        b"fLaC": ".flac",
        b"OggS": ".ogg",
        b"ID3": ".mp3",
        b"GIF8": ".gif",
    }
    for k, v in sig.items():
        if b.startswith(k):
            if k == b"RIFF" and b[8:12] == b"WEBP":
                return ".webp"
            if k == b"RIFF" and b[8:12] == b"AVI ":
                return ".avi"
            return v
    if b[4:8] == b"ftyp":
        return ".mp4"
    if b.startswith(b"\x1aE\xdf\xa3"):
        return ".webm"
    return None


def arrange_folders(src: Path, raw: Path, link: bool, dry: bool) -> dict:
    stats = {"copied": 0, "skipped_no_label": 0, "skipped_unknown_type": 0}
    for f in src.rglob("*"):
        if not f.is_file() or raw in f.parents:
            continue
        mod = modality_from_extension(f)
        if mod is None:
            if f.suffix.lower() not in {".md", ".json", ".txt", ".csv", ".parquet", ".arrow", ".gitattributes", ""}:
                stats["skipped_unknown_type"] += 1
            continue
        label, source = infer_label_and_source(f.relative_to(src).parts)
        if label is None:
            stats["skipped_no_label"] += 1
            continue
        target = raw / mod / ("fake" if label else "real") / _safe(source) / f.name
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                (target.symlink_to(f.resolve()) if link else shutil.copy2(f, target))
        stats["copied"] += 1
    return stats


def _pick(cols: list[str], hints: tuple[str, ...]) -> str | None:
    low = {c.lower(): c for c in cols}
    for h in hints:
        for lc, c in low.items():
            if lc == h:
                return c
    for h in hints:
        for lc, c in low.items():
            if h in lc:
                return c
    return None


def arrange_parquet(src: Path, raw: Path, dry: bool) -> dict:
    stats = {"exported": 0, "skipped_no_label": 0, "skipped_no_media": 0, "files": 0}
    for pq in sorted(list(src.rglob("*.parquet"))):
        df = pd.read_parquet(pq)
        stats["files"] += 1
        cols = list(df.columns)
        lab_c, src_c = _pick(cols, LABEL_HINTS), _pick(cols, SOURCE_HINTS)
        media_c = next((c for c in cols if df[c].map(lambda v: isinstance(v, (dict, bytes))).mean() > 0.9), None)
        mod_hint = next(
            (m for m in ("image", "audio", "video") if m in pq.name.lower() or m in str(pq.parent).lower()), None
        )
        print(
            f"  {pq.relative_to(src)}: rows={len(df)} label={lab_c} source={src_c} media={media_c} modality_hint={mod_hint}"
        )
        if media_c is None:
            stats["skipped_no_media"] += len(df)
            continue
        for i, row in df.iterrows():
            v = row[media_c]
            blob, name = (v.get("bytes"), v.get("path")) if isinstance(v, dict) else (v, None)
            if not blob:
                stats["skipped_no_media"] += 1
                continue
            label = _norm_label(row[lab_c]) if lab_c else None
            if label is None and lab_c and isinstance(row[lab_c], str):
                toks = set(row[lab_c].lower().replace("-", "_").split("_"))
                label = 1 if toks & FAKE_TOKENS else 0 if toks & REAL_TOKENS else None
            if label is None:
                stats["skipped_no_label"] += 1
                continue
            ext = (Path(name).suffix.lower() if name else "") or _sniff_ext(blob) or ""
            mod = modality_from_extension(f"x{ext}") or mod_hint
            if mod is None or not ext:
                stats["skipped_no_media"] += 1
                continue
            sv = row[src_c] if src_c else None
            source = (
                _safe(sv)
                if (sv is not None and not pd.isna(sv) and str(sv).strip())
                else ("real" if label == 0 else "unknown")
            )
            target = raw / mod / ("fake" if label else "real") / source / f"{pq.stem}_{i:06d}{ext}"
            if not dry:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
            stats["exported"] += 1
    return stats


def summary(raw: Path) -> None:
    from forensix.data.dataset import scan_raw

    if not raw.exists():
        print("nothing arranged yet")
        return
    df = scan_raw(raw)
    if df.empty:
        print("\n!! no labelled media found under", raw)
        return
    print("\n=== arranged dataset ===")
    print(df.groupby(["modality", "label"]).size().unstack(fill_value=0).rename(columns={0: "real", 1: "fake"}))
    print("\nsources per modality:")
    print(df[df["label"] == 1].groupby("modality")["source"].nunique())
    print("\nExpected from the synopsis: images 2000/2000, audio 1000/1000, video 100/100. Compare the table above.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, help="already-downloaded copy (skip the download)")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "_download")
    ap.add_argument("--link", action="store_true", help="symlink instead of copy (saves disk space)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = a.src or download(a.cache, a.repo)
    print("source:", src)
    f = arrange_folders(src, a.raw, a.link, a.dry_run)
    print("folder-based:", f)
    if list(src.rglob("*.parquet")):
        print("parquet files found:")
        print("parquet-based:", arrange_parquet(src, a.raw, a.dry_run))
    if not a.dry_run:
        summary(a.raw)
        print("\nNext:  make pipeline")


if __name__ == "__main__":
    main()
