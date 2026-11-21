#!/usr/bin/env python3
"""List the model registry or approve a version for production.

python scripts/registry_cli.py list
python scripts/registry_cli.py promote image xgboost 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forensix.config import get_config  # noqa: E402
from forensix.mlops.registry import ModelRegistry  # noqa: E402

reg = ModelRegistry(get_config())
if len(sys.argv) >= 5 and sys.argv[1] == "promote":
    print(
        reg.set_stage(sys.argv[2], sys.argv[3], int(sys.argv[4]), "production", "approved via CLI")["display"],
        "-> production",
    )
else:
    for m in reg.all():
        print(
            f"{m['display']:32s} {m['stage']:11s} val_f1={m['metrics'].get('val_f1', float('nan')):.3f} test_f1={m['metrics'].get('test_f1', float('nan')):.3f}"
        )
