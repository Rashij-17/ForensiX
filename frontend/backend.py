"""Backend selection for the Streamlit UI.

* If ``FORENSIX_API_URL`` (default ``http://localhost:8000``) answers ``/health`` the UI talks to the FastAPI backend.
* Otherwise it falls back to running the same service layer in-process, so the UI also works stand-alone
  (``streamlit run frontend/app.py``) without starting the API first.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "sdk"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import requests  # noqa: E402

from forensix_client import ForensiXClient  # noqa: E402


class RemoteBackend:
    mode = "api"

    def __init__(self, url: str):
        self.url = url
        self.c = ForensiXClient(url)

    def __getattr__(self, name):  # status, benchmarks, monitoring, cases ... map 1:1
        return getattr(self.c, name)

    def compare(self, a, na, b, nb):
        return self.c.compare(a, b, na, nb)

    def lab(self, data, name, kind, strengths=None):
        return self.c.lab(data, kind, strengths, name=name)


class LocalBackend:
    mode = "local"

    def __init__(self):
        from forensix.services import ForensiXService

        self.s = ForensiXService(os.environ.get("FORENSIX_HOME"))

    def __getattr__(self, name):
        return getattr(self.s, name)


def get_backend():
    url = os.environ.get("FORENSIX_API_URL", "http://localhost:8000")
    try:
        if requests.get(f"{url}/health", timeout=1.5).ok:
            return RemoteBackend(url)
    except Exception:
        pass
    return LocalBackend()
