"""Dataset construction: scan raw files, validate, deduplicate, leak-free splits, metadata table, features."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..features import extract_from_path
from ..preprocess import audio as pa
from ..preprocess import image as pi
from ..preprocess import video as pv
from ..utils import get_logger, md5_file, modality_from_extension

log = get_logger()

REAL_TOKENS = {"real", "authentic", "bonafide", "genuine", "original", "pristine", "natural", "live"}
FAKE_TOKENS = {"fake", "manipulated", "spoof", "spoofed", "synthetic", "deepfake", "generated", "forged", "tampered"}

META_COLUMNS = [
    "sample_id",
    "modality",
    "label",
    "label_name",
    "source",
    "file_format",
    "width",
    "height",
    "duration_s",
    "fps",
    "path",
    "md5",
    "status",
    "status_reason",
    "dup_group",
    "split",
    "preprocessing_version",
]


def _tokens(part: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", part.lower())) - {""}


def infer_label_and_source(rel_parts: tuple[str, ...]) -> tuple[int | None, str]:
    """Infer (label, source) from directory names, e.g. ``image/fake/stable_diffusion/x.png``.

    The first directory containing a real/fake token decides the label; the directory right below it (if any)
    is the source / generator. Real samples without a sub-directory get source ``real``.
    """
    dirs = rel_parts[:-1]
    for i, part in enumerate(dirs):
        toks = _tokens(part)
        if toks & FAKE_TOKENS and not toks & REAL_TOKENS:
            src = dirs[i + 1] if i + 1 < len(dirs) else "unknown"
            return 1, src
        if toks & REAL_TOKENS and not toks & FAKE_TOKENS:
            src = dirs[i + 1] if i + 1 < len(dirs) else "real"
            return 0, src
    return None, "unknown"


def _norm_label(v) -> int | None:
    s = str(v).strip().lower()
    if s in {"1", "fake", "manipulated", "spoof", "synthetic", "deepfake", "generated", "true"}:
        return 1
    if s in {"0", "real", "authentic", "bonafide", "genuine", "original", "false"}:
        return 0
    return None


def scan_raw(raw_dir: Path) -> pd.DataFrame:
    """Walk ``raw_dir`` and return one row per recognised media file with label & source.

    An optional ``metadata.csv`` (columns ``path,label[,source]``; path relative to ``raw_dir``) overrides the
    folder-based inference - use it when the dataset does not encode labels in directory names.
    """
    override = None
    meta_csv = raw_dir / "metadata.csv"
    if meta_csv.exists():
        m = pd.read_csv(meta_csv)
        m.columns = [c.strip().lower() for c in m.columns]
        if {"path", "label"} <= set(m.columns):
            override = {
                str(Path(r["path"]).as_posix()): (_norm_label(r["label"]), str(r.get("source", "unknown")))
                for _, r in m.iterrows()
            }
    rows = []
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file():
            continue
        modality = modality_from_extension(p)
        if modality is None:
            continue
        rel = p.relative_to(raw_dir)
        if override is not None and rel.as_posix() in override:
            label, source = override[rel.as_posix()]
        else:
            label, source = infer_label_and_source(rel.parts)
        if label is None:
            log.warning("skipping %s: cannot infer real/fake label (use directory names or metadata.csv)", rel)
            continue
        rows.append(
            {
                "modality": modality,
                "label": int(label),
                "source": source or "unknown",
                "abs_path": str(p),
                "path": rel.as_posix(),
                "file_format": p.suffix.lower().lstrip("."),
            }
        )
    return pd.DataFrame(rows)


def _inspect(row: dict, cfg) -> dict:
    """Validate one file and collect its technical metadata (corruption / silence / duplicate fingerprint)."""
    out = {
        "md5": "",
        "width": np.nan,
        "height": np.nan,
        "duration_s": np.nan,
        "fps": np.nan,
        "status": "ok",
        "status_reason": "ok",
        "dhash": "",
    }
    path, modality = row["abs_path"], row["modality"]
    try:
        out["md5"] = md5_file(path)
        if modality == "image":
            rgb = pi.load_image(path)
            if rgb is None:
                return {**out, "status": "invalid", "status_reason": "corrupt_image"}
            out["height"], out["width"] = rgb.shape[:2]
            out["dhash"] = pi.dhash(rgb)
        elif modality == "audio":
            a = cfg["audio"]
            y = pa.load_audio(path, a["sr"])
            ok, reason = pa.validate_audio(y, a["sr"], a["min_seconds"], a["silence_db"])
            if y is not None:
                out["duration_s"] = round(y.size / a["sr"], 3)
            if not ok:
                return {**out, "status": "invalid", "status_reason": reason}
        else:
            info = pv.probe_video(path)
            if info is None:
                return {**out, "status": "invalid", "status_reason": "corrupt_video"}
            out.update(
                width=info["width"],
                height=info["height"],
                duration_s=round(info["duration"], 3),
                fps=round(info["fps"], 3),
            )
    except Exception as exc:  # pragma: no cover - defensive
        return {**out, "status": "invalid", "status_reason": f"error:{type(exc).__name__}"}
    return out


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def assign_duplicate_groups(df: pd.DataFrame, near_hamming: int = 2) -> pd.Series:
    """Group exact duplicates (md5) and near-duplicate images (dHash Hamming <= ``near_hamming``)."""
    parent = {i: i for i in df.index}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ok = df[df["status"] == "ok"]
    for _, grp in ok.groupby(["modality", "md5"]):
        idx = list(grp.index)
        for j in idx[1:]:
            union(idx[0], j)
    imgs = ok[(ok["modality"] == "image") & (ok["dhash"] != "")]
    if len(imgs) > 1:
        # exhaustive pairwise Hamming distance on the 64-bit hashes (vectorised; no bucketing that could miss pairs)
        idx = imgs.index.to_numpy()
        bits = np.array([[c == "1" for c in h] for h in imgs["dhash"]], dtype=np.uint8)
        for a in range(len(idx) - 1):
            close = np.nonzero((bits[a + 1 :] != bits[a]).sum(axis=1) <= near_hamming)[0]
            for b in close:
                union(idx[a], idx[a + 1 + b])
    return pd.Series({i: f"g{find(i)}" for i in df.index})


def assign_splits(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.Series:
    """Leak-free train/val/test assignment.

    Every duplicate group lives entirely in one split. Within each (modality, label) stratum, whole groups are
    shuffled and filled into train / val / test so that ratios stay close to the requested fractions.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("excluded", index=df.index, dtype=object)
    names = ["train", "val", "test"]
    fr = np.array([fractions[n] for n in names], dtype=float)
    fr = fr / fr.sum()
    ok = df[df["status"] == "ok"]
    for (_, _label), stratum in ok.groupby(["modality", "label"]):
        groups = stratum.groupby("dup_group").size()
        order = rng.permutation(len(groups))
        gids, sizes = groups.index.to_numpy()[order], groups.to_numpy()[order]
        total = sizes.sum()
        targets = fr * total
        filled = np.zeros(3)
        assign = {}
        for gid, sz in zip(gids, sizes):
            deficit = targets - filled  # largest remaining need first
            k = int(np.argmax(deficit))
            assign[gid] = names[k]
            filled[k] += sz
        for i in stratum.index:
            split.at[i] = assign[df.at[i, "dup_group"]]
    return split


def build_metadata(cfg, n_jobs: int = -1, limit_per_class: int | None = None) -> pd.DataFrame:
    """Stage 1+2: scan, validate, deduplicate, split. Writes ``data/processed/metadata.csv``."""
    scan = scan_raw(cfg.raw_dir)
    if scan.empty:
        raise FileNotFoundError(
            f"No labelled media found under {cfg.raw_dir}. Place the DeepSafe files there "
            "(see docs/DATASET.md) or run `make demo-data` to generate a small synthetic demo set."
        )
    if limit_per_class:
        scan = scan.groupby(["modality", "label"], group_keys=False).head(limit_per_class).reset_index(drop=True)
    log.info(
        "scanned %d files (%s)", len(scan), ", ".join(f"{m}: {n}" for m, n in scan["modality"].value_counts().items())
    )
    infos = Parallel(n_jobs=n_jobs)(delayed(_inspect)(r, cfg) for r in scan.to_dict("records"))
    df = pd.concat([scan.reset_index(drop=True), pd.DataFrame(infos)], axis=1)
    df["dup_group"] = assign_duplicate_groups(df)
    # Only the first member of every duplicate group is kept; the rest are flagged and excluded, so exact and
    # near duplicates can never straddle the train / val / test boundary.
    conflict = df[df["status"] == "ok"].groupby(["modality", "dup_group"])["label"].nunique()
    conflicting = {g for (_, g), n in conflict.items() if n > 1}
    later = (df["status"] == "ok") & df.duplicated(["modality", "dup_group"], keep="first")
    df.loc[later, ["status", "status_reason"]] = ["duplicate", "duplicate_of_earlier_sample"]
    clash = df["dup_group"].isin(conflicting) & (df["status"] == "ok")
    df.loc[clash, ["status", "status_reason"]] = ["invalid", "label_conflict_with_duplicate"]
    df["split"] = assign_splits(df, cfg["data"]["splits"], cfg["seed"])
    df["preprocessing_version"] = cfg["versions"]["preprocessing"]
    df["label_name"] = df["label"].map({0: "real", 1: "fake"})
    df["sample_id"] = [f"{m[:3]}_{i:05d}" for i, m in enumerate(df["modality"])]
    out = df[META_COLUMNS + ["abs_path"]].copy()
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["abs_path"]).to_csv(cfg.metadata_path, index=False)
    out[["sample_id", "abs_path"]].to_csv(cfg.processed_dir / "paths.csv", index=False)
    summarize_metadata(out, log.info)
    return out


def summarize_metadata(df: pd.DataFrame, emit=print) -> None:
    for m, g in df.groupby("modality"):
        c = g["status"].value_counts().to_dict()
        s = g[g["status"] == "ok"]["split"].value_counts().to_dict()
        emit(f"[{m}] files={len(g)} status={c} splits={s}")


def load_metadata(cfg) -> pd.DataFrame:
    if not cfg.metadata_path.exists():
        raise FileNotFoundError("metadata.csv missing - run the ingest stage first (`make ingest`).")
    df = pd.read_csv(cfg.metadata_path)
    paths = cfg.processed_dir / "paths.csv"
    if paths.exists():
        df = df.merge(pd.read_csv(paths), on="sample_id", how="left")
    else:
        df["abs_path"] = df["path"].map(lambda p: str(cfg.raw_dir / p))
    return df


def build_features(cfg, modality: str, n_jobs: int = -1) -> pd.DataFrame:
    """Stage 3: extract forensic features for every usable sample of one modality."""
    meta = load_metadata(cfg)
    todo = meta[(meta["modality"] == modality) & (meta["status"] == "ok") & (meta["split"] != "excluded")]
    if todo.empty:
        raise RuntimeError(f"no usable {modality} samples in metadata")
    log.info("[%s] extracting features for %d samples", modality, len(todo))
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_from_path)(r["abs_path"], modality, cfg) for r in todo.to_dict("records")
    )
    rows, failed = [], []
    for sid, feats in zip(todo["sample_id"], results):
        if feats is None:
            failed.append(sid)
        else:
            rows.append({"sample_id": sid, **feats})
    if failed:
        meta.loc[meta["sample_id"].isin(failed), ["status", "status_reason", "split"]] = [
            "invalid",
            "feature_extraction_failed",
            "excluded",
        ]
        meta.drop(columns=["abs_path"], errors="ignore").to_csv(cfg.metadata_path, index=False)
        log.warning("[%s] %d samples failed extraction and were excluded", modality, len(failed))
    feats = pd.DataFrame(rows)
    feats.to_csv(cfg.features_path(modality), index=False)
    log.info(
        "[%s] wrote %s (%d rows x %d features)",
        modality,
        cfg.features_path(modality).name,
        len(feats),
        feats.shape[1] - 1,
    )
    return feats


def load_feature_table(cfg, modality: str) -> pd.DataFrame:
    """Features joined with label / source / split for one modality (usable samples only)."""
    meta = load_metadata(cfg)
    f = pd.read_csv(cfg.features_path(modality))
    cols = ["sample_id", "label", "source", "split", "path"]
    return f.merge(meta[cols], on="sample_id", how="inner")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"sample_id", "label", "source", "split", "path"}]
