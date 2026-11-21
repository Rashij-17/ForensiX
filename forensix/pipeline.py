"""The twelve-stage pipeline, runnable stage by stage or end to end."""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from .config import MODALITIES, get_config
from .utils import get_logger, quiet_warnings

log = get_logger()

STAGES = [
    "demo-data",
    "ingest",
    "features",
    "eda",
    "stats",
    "train",
    "explain",
    "robustness",
    "generators",
    "mlops-init",
    "summary",
]


def _mods(args) -> list[str]:
    return [args.modality] if args.modality and args.modality != "all" else list(MODALITIES)


def stage_demo_data(cfg, args):
    from .data.synthetic import generate_demo_dataset

    counts = generate_demo_dataset(
        cfg.raw_dir, n_image=args.demo_images, n_audio=args.demo_audio, n_video=args.demo_videos
    )
    log.info("generated SYNTHETIC demo data in %s: %s (not the DeepSafe dataset)", cfg.raw_dir, counts)


def stage_ingest(cfg, args):
    from .data import build_metadata
    from .mlops.versioning import record_version

    meta = build_metadata(cfg, n_jobs=args.jobs, limit_per_class=args.limit)
    v = record_version(cfg, meta, note="ingest")
    log.info("dataset version %s: %s samples", v["version"], v["n_samples"])


def stage_features(cfg, args):
    from .data import build_features

    for m in _mods(args):
        if cfg.metadata_path.exists():
            build_features(cfg, m, n_jobs=args.jobs)


def stage_eda(cfg, args):
    from .eda import run_eda

    for m in _mods(args):
        run_eda(cfg, m)


def stage_stats(cfg, args):
    from .stats import run_statistics

    for m in _mods(args):
        run_statistics(cfg, m)


def stage_train(cfg, args):
    from .models import train_modality

    for m in _mods(args):
        train_modality(cfg, m)


def stage_explain(cfg, args):
    from .explain.global_report import run_global_explanations

    for m in _mods(args):
        run_global_explanations(cfg, m)


def stage_robustness(cfg, args):
    from .robustness import run_robustness

    for m in _mods(args):
        run_robustness(cfg, m, n_jobs=args.jobs, max_samples=args.limit_test)


def stage_generators(cfg, args):
    from .models.generators import run_generator_analysis

    for m in _mods(args):
        run_generator_analysis(cfg, m)


def stage_mlops_init(cfg, args):
    from .mlops.registry import ModelRegistry

    reg = ModelRegistry(cfg)
    for m in _mods(args):
        p = reg.production(m)
        log.info("[%s] production model: %s", m, p["display"] if p else "none")


def stage_summary(cfg, args):
    from .models import comparison_table

    t = comparison_table(cfg)
    if not t.empty:
        pd.set_option("display.width", 200)
        print(t.round(3).to_string(index=False))
        t.to_csv(cfg.reports_dir / "model_comparison.csv", index=False)


_FUNCS = {
    "demo-data": stage_demo_data,
    "ingest": stage_ingest,
    "features": stage_features,
    "eda": stage_eda,
    "stats": stage_stats,
    "train": stage_train,
    "explain": stage_explain,
    "robustness": stage_robustness,
    "generators": stage_generators,
    "mlops-init": stage_mlops_init,
    "summary": stage_summary,
}

DEFAULT_RUN = ["ingest", "features", "eda", "stats", "train", "explain", "robustness", "generators", "summary"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="ForensiX pipeline")
    ap.add_argument("stages", nargs="*", help=f"stages to run ({', '.join(STAGES)}); default: full pipeline")
    ap.add_argument("--modality", default="all", choices=["all", *MODALITIES])
    ap.add_argument("--root", default=None, help="project root (default: repo root or $FORENSIX_HOME)")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    ap.add_argument("--limit", type=int, default=None, help="max files per (modality, class) at ingest (debug)")
    ap.add_argument("--limit-test", type=int, default=None, help="max test samples per modality for robustness")
    ap.add_argument("--demo-images", type=int, default=160)
    ap.add_argument("--demo-audio", type=int, default=80)
    ap.add_argument("--demo-videos", type=int, default=36)
    args = ap.parse_args(argv)
    quiet_warnings()
    cfg = get_config(args.root)
    stages = args.stages or DEFAULT_RUN
    unknown = [s for s in stages if s not in _FUNCS]
    if unknown:
        ap.error(f"unknown stage(s): {unknown}")
    for s in stages:
        t = time.time()
        log.info("=== stage: %s ===", s)
        _FUNCS[s](cfg, args)
        log.info("stage %s finished in %.1fs", s, time.time() - t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
