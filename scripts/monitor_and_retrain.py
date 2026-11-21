#!/usr/bin/env python3
"""Check drift / degradation for each modality and retrain when triggered (suitable for cron / CI).

python scripts/monitor_and_retrain.py            # check and retrain if triggered
python scripts/monitor_and_retrain.py --check    # only report
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forensix.config import MODALITIES, get_config  # noqa: E402
from forensix.mlops.monitoring import retrain_decision, run_retraining  # noqa: E402
from forensix.utils import json_safe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
cfg = get_config()
for m in MODALITIES:
    d = retrain_decision(cfg, m)
    print(f"[{m}] retrain={d['retrain']} logged={d['monitoring']['n_logged']} reasons={d['reasons']}")
    if not a.check and (d["retrain"] or a.force):
        print(
            json.dumps(
                json_safe({k: v for k, v in run_retraining(cfg, m, force=a.force).items() if k != "decision"}), indent=2
            )
        )
