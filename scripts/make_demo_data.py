#!/usr/bin/env python3
"""Generate the SYNTHETIC demo dataset (not DeepSafe) so the whole pipeline can be exercised offline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forensix.config import get_config  # noqa: E402
from forensix.data.synthetic import generate_demo_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--images", type=int, default=240)
ap.add_argument("--audio", type=int, default=120)
ap.add_argument("--videos", type=int, default=60)
a = ap.parse_args()
cfg = get_config()
print(generate_demo_dataset(cfg.raw_dir, a.images, a.audio, a.videos), "->", cfg.raw_dir)
