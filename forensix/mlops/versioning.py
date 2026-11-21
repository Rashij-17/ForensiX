"""Dataset versioning: successive dataset revisions with sample counts, preprocessing / feature versions, date.

``artifacts/dataset_versions.json`` is a plain, diff-friendly log. It works on its own and complements DVC
(``dvc.yaml`` tracks the actual files; this log records what each revision *contained* and how it differs).
"""

from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd

from ..utils import read_json, write_json


def _path(cfg):
    return cfg.artifacts_dir / "dataset_versions.json"


def list_versions(cfg) -> list[dict]:
    return read_json(_path(cfg), {"versions": []})["versions"]


def current_version(cfg) -> str:
    v = list_versions(cfg)
    return v[-1]["version"] if v else "v0"


def _fingerprint(meta: pd.DataFrame, extra: list[str] | None = None) -> str:
    ok = meta[(meta["status"] == "ok") & (meta["split"] != "excluded")]
    h = hashlib.sha256()
    for token in sorted(ok["md5"].astype(str) + ":" + ok["split"].astype(str) + ":" + ok["label"].astype(str)):
        h.update(token.encode())
    for token in sorted(extra or []):
        h.update(token.encode())
    return h.hexdigest()[:16]


def record_version(
    cfg, meta: pd.DataFrame, note: str = "", n_verified_added: int = 0, extra_fingerprint: list[str] | None = None
) -> dict:
    """Append a dataset revision unless nothing changed since the last one (same fingerprint)."""
    versions = list_versions(cfg)
    fp = _fingerprint(meta, extra_fingerprint)
    if versions and versions[-1]["fingerprint"] == fp:
        return versions[-1]
    ok = meta[(meta["status"] == "ok") & (meta["split"] != "excluded")]
    counts = {m: int((ok["modality"] == m).sum()) for m in ("image", "audio", "video")}
    entry = {
        "version": f"v{len(versions) + 1}",
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fp,
        "n_samples": counts,
        "n_total": int(sum(counts.values())),
        "n_files_scanned": int(len(meta)),
        "n_excluded": int((meta["status"] != "ok").sum()),
        "splits": {s: int((ok["split"] == s).sum()) for s in ("train", "val", "test")},
        "preprocessing_version": cfg["versions"]["preprocessing"],
        "feature_version": cfg["versions"]["feature"],
        "verified_production_samples_added": int(n_verified_added),
        "note": note or ("initial dataset" if not versions else "dataset changed"),
    }
    versions.append(entry)
    write_json(_path(cfg), {"versions": versions})
    return entry
