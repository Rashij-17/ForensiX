"""Tiny Python client for the ForensiX API (also used by the Streamlit frontend in remote mode).

from forensix_client import ForensiXClient
c = ForensiXClient("http://localhost:8000")
result = c.analyze("photo.jpg")
print(result["verdict"]["text"], result["probability_manipulated"])
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

import requests


class ForensiXError(RuntimeError):
    pass


class ForensiXClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 300.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # ---- plumbing -------------------------------------------------------------------
    def _req(self, method: str, path: str, **kw):
        r = requests.request(method, f"{self.base}{path}", timeout=self.timeout, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise ForensiXError(str(detail))
        return r.json()

    @staticmethod
    def _file(data_or_path, name: str | None = None):
        if isinstance(data_or_path, (str, Path)):
            p = Path(data_or_path)
            return (name or p.name, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        return (name or "upload", data_or_path, "application/octet-stream")

    # ---- core -------------------------------------------------------------------------
    def health(self):
        return self._req("GET", "/health")

    def status(self):
        return self._req("GET", "/status")

    def analyze(self, file, name: str | None = None, case_id: str | None = None, explain: bool = True, note: str = ""):
        return self._req(
            "POST",
            "/analyze",
            files={"file": self._file(file, name)},
            data={"case_id": case_id or "", "explain": str(explain).lower(), "note": note},
        )

    def evidence(self, file, name: str | None = None):
        return self._req("POST", "/evidence", files={"file": self._file(file, name)})

    def compare(self, file_a, file_b, name_a: str | None = None, name_b: str | None = None):
        return self._req(
            "POST", "/compare", files={"file_a": self._file(file_a, name_a), "file_b": self._file(file_b, name_b)}
        )

    def lab_options(self, modality: str):
        return self._req("GET", f"/lab/options/{modality}")

    def lab(self, file, kind: str, strengths: list[float] | None = None, name: str | None = None):
        return self._req(
            "POST",
            "/lab",
            files={"file": self._file(file, name)},
            data={"kind": kind, "strengths": ",".join(str(s) for s in strengths or [])},
        )

    def report(self, file, name: str | None = None, case_id: str | None = None, analyst: str = ""):
        return self._req(
            "POST",
            "/report",
            files={"file": self._file(file, name)},
            data={"case_id": case_id or "", "analyst": analyst},
        )

    # ---- cases -------------------------------------------------------------------------
    def create_case(self, name: str, description: str = "", analyst: str = ""):
        return self._req("POST", "/cases", json={"name": name, "description": description, "analyst": analyst})

    def list_cases(self):
        return self._req("GET", "/cases")

    def get_case(self, case_id: str):
        return self._req("GET", f"/cases/{case_id}")

    def set_case_status(self, case_id: str, status: str):
        return self._req("PATCH", f"/cases/{case_id}/status", json={"status": status})

    def add_case_note(self, case_id: str, text: str):
        return self._req("POST", f"/cases/{case_id}/notes", json={"text": text})

    def case_report(self, case_id: str):
        return self._req("GET", f"/cases/{case_id}/report")

    def delete_case(self, case_id: str):
        return self._req("DELETE", f"/cases/{case_id}")

    # ---- benchmarks / MLOps ---------------------------------------------------------------
    def benchmarks(self):
        return self._req("GET", "/benchmarks")

    def feature_stats(self, modality: str):
        return self._req("GET", f"/benchmarks/stats/{modality}")

    def registry_list(self):
        return self._req("GET", "/registry")

    def promote(self, modality: str, algorithm: str, version: int):
        return self._req(
            "POST", "/registry/promote", json={"modality": modality, "algorithm": algorithm, "version": version}
        )

    def monitoring(self, modality: str):
        return self._req("GET", f"/monitoring/{modality}")

    def prediction_log(self, limit: int = 200):
        return self._req("GET", "/predictions", params={"limit": limit})

    def feedback(self, record_id: str, label: int):
        return self._req("POST", "/feedback", json={"record_id": record_id, "label": label})

    def retrain_check(self, modality: str):
        return self._req("GET", f"/retrain/{modality}/check")

    def retrain(self, modality: str, force: bool = False):
        return self._req("POST", f"/retrain/{modality}", json={"force": force})

    def experiments(self, limit: int = 200):
        return self._req("GET", "/experiments", params={"limit": limit})

    def dataset_versions(self):
        return self._req("GET", "/dataset/versions")
