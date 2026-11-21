#!/usr/bin/env python3
"""Check the API and (optionally) analyse one file.   python scripts/health_check.py [file]"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))
from forensix_client import ForensiXClient  # noqa: E402

c = ForensiXClient(sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000")
print("health:", c.health())
if len(sys.argv) > 1:
    r = c.analyze(sys.argv[1], explain=False)
    print(json.dumps({k: r[k] for k in ("modality", "probability_manipulated", "verdict", "model")}, indent=2))
