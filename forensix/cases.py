"""Investigations: group several analyses into a case and track them together (SQLite, no media stored)."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from contextlib import closing

from .utils import json_safe

STATUSES = ("open", "under review", "closed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT, analyst TEXT, status TEXT NOT NULL,
  created TEXT NOT NULL, updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE, record_id TEXT,
  filename TEXT, sha256 TEXT, modality TEXT, probability REAL, verdict TEXT, model TEXT, created TEXT NOT NULL,
  note TEXT, summary_json TEXT);
CREATE TABLE IF NOT EXISTS case_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  ts TEXT NOT NULL, event TEXT NOT NULL);
"""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class CaseStore:
    def __init__(self, cfg):
        self.path = cfg.artifacts_dir / "cases.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        return c

    @staticmethod
    def _event(c, case_id: str, text: str) -> None:
        c.execute("INSERT INTO case_events (case_id, ts, event) VALUES (?, ?, ?)", (case_id, _now(), text))

    # ---- cases ---------------------------------------------------------------------
    def create(self, name: str, description: str = "", analyst: str = "") -> dict:
        if not name.strip():
            raise ValueError("a case needs a name")
        cid = "CASE-" + uuid.uuid4().hex[:8].upper()
        now = _now()
        with closing(self._conn()) as c:
            c.execute(
                "INSERT INTO cases VALUES (?,?,?,?,?,?,?)", (cid, name.strip(), description, analyst, "open", now, now)
            )
            self._event(c, cid, "case opened")
            c.commit()
        return self.get(cid)

    def list(self) -> list[dict]:
        with closing(self._conn()) as c:
            rows = c.execute("""SELECT c.*, COUNT(a.id) AS n_analyses,
                                       SUM(CASE WHEN a.verdict='manipulated' THEN 1 ELSE 0 END) AS n_flagged
                                FROM cases c LEFT JOIN analyses a ON a.case_id = c.id
                                GROUP BY c.id ORDER BY c.updated DESC""").fetchall()
        return [{**dict(r), "n_flagged": r["n_flagged"] or 0} for r in rows]

    def get(self, case_id: str) -> dict:
        with closing(self._conn()) as c:
            case = c.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
            if case is None:
                raise KeyError(case_id)
            an = c.execute("SELECT * FROM analyses WHERE case_id=? ORDER BY created", (case_id,)).fetchall()
            ev = c.execute("SELECT ts, event FROM case_events WHERE case_id=? ORDER BY id", (case_id,)).fetchall()
        analyses = []
        for r in an:
            d = dict(r)
            d["summary"] = json.loads(d.pop("summary_json") or "{}")
            analyses.append(d)
        probs = [a["probability"] for a in analyses if a["probability"] is not None]
        return {
            **dict(case),
            "analyses": analyses,
            "events": [dict(e) for e in ev],
            "stats": {
                "n_analyses": len(analyses),
                "n_flagged": sum(a["verdict"] == "manipulated" for a in analyses),
                "n_inconclusive": sum(a["verdict"] == "inconclusive" for a in analyses),
                "mean_probability": sum(probs) / len(probs) if probs else None,
                "max_probability": max(probs) if probs else None,
            },
        }

    def set_status(self, case_id: str, status: str) -> dict:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        with closing(self._conn()) as c:
            if not c.execute("UPDATE cases SET status=?, updated=? WHERE id=?", (status, _now(), case_id)).rowcount:
                raise KeyError(case_id)
            self._event(c, case_id, f"status set to '{status}'")
            c.commit()
        return self.get(case_id)

    def delete(self, case_id: str) -> None:
        with closing(self._conn()) as c:
            c.execute("DELETE FROM cases WHERE id=?", (case_id,))
            c.commit()

    # ---- analyses -------------------------------------------------------------------
    def add_analysis(self, case_id: str, result: dict, note: str = "") -> dict:
        """Attach a (media-free) analysis summary to a case."""
        summary = {
            "model": result["model"]["display"],
            "verdict_text": result["verdict"]["text"],
            "evidence_strength": result["verdict"]["evidence_strength"],
            "top_indicators": [
                {"feature": c["feature"], "group": c["group"], "direction": c["direction"], "shap": c["shap"]}
                for c in result.get("explanation", {}).get("contributions", [])[:5]
            ],
            "elapsed_s": result.get("elapsed_s"),
        }
        aid = uuid.uuid4().hex[:10]
        with closing(self._conn()) as c:
            if c.execute("SELECT 1 FROM cases WHERE id=?", (case_id,)).fetchone() is None:
                raise KeyError(case_id)
            c.execute(
                "INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    aid,
                    case_id,
                    result.get("record_id"),
                    result.get("filename"),
                    result.get("sha256"),
                    result["modality"],
                    result["probability_manipulated"],
                    result["verdict"]["label"],
                    result["model"]["display"],
                    _now(),
                    note,
                    json.dumps(json_safe(summary)),
                ),
            )
            c.execute("UPDATE cases SET updated=? WHERE id=?", (_now(), case_id))
            self._event(
                c,
                case_id,
                f"analysed {result.get('filename')} -> {result['verdict']['text']} "
                f"(P={result['probability_manipulated']:.2f})",
            )
            c.commit()
        return self.get(case_id)

    def add_note(self, case_id: str, text: str) -> dict:
        with closing(self._conn()) as c:
            self._event(c, case_id, f"note: {text}")
            c.execute("UPDATE cases SET updated=? WHERE id=?", (_now(), case_id))
            c.commit()
        return self.get(case_id)
