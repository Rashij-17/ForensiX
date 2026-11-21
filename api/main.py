"""ForensiX FastAPI backend.

Run:  uvicorn api.main:app --port 8000      Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from forensix import __version__
from forensix.services import ForensiXService
from forensix.utils import quiet_warnings

quiet_warnings()
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["svc"] = ForensiXService(os.environ.get("FORENSIX_HOME"))
    yield
    state.clear()


app = FastAPI(
    title="ForensiX API",
    version=__version__,
    lifespan=lifespan,
    description="Multimodal digital-media forensics: authenticity probability, evidence, SHAP explanations, "
    "stress tests, cases, reports and MLOps controls. Output is decision support, not proof.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def svc() -> ForensiXService:
    return state["svc"]


async def _read(upload: UploadFile) -> tuple[bytes, str]:
    data = await upload.read()
    if not data:
        raise HTTPException(400, "empty file")
    return data, upload.filename or "upload"


def _guard(fn, *a, **kw):
    """Map expected errors to clean HTTP responses."""
    try:
        return fn(*a, **kw)
    except KeyError as e:
        raise HTTPException(404, f"not found: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))


# ------------------------------------------------------------------ health / status
@app.get("/health", tags=["system"])
def health():
    s = svc().status()
    return {"status": "ok" if s["ready"] else "no_models", "version": __version__, "modalities": s["modalities"]}


@app.get("/status", tags=["system"])
def status():
    return svc().status()


# ------------------------------------------------------------------ analyze / evidence
@app.post("/analyze", tags=["analyze"])
async def analyze(
    file: UploadFile = File(...), case_id: str | None = Form(None), explain: bool = Form(True), note: str = Form("")
):
    data, name = await _read(file)
    return _guard(svc().analyze, data, name, case_id or None, explain, note)


@app.post("/evidence", tags=["analyze"])
async def evidence(file: UploadFile = File(...)):
    data, name = await _read(file)
    return _guard(svc().evidence, data, name)


# ------------------------------------------------------------------ compare / lab / report
@app.post("/compare", tags=["compare"])
async def compare(file_a: UploadFile = File(...), file_b: UploadFile = File(...)):
    (da, na), (db, nb) = await _read(file_a), await _read(file_b)
    return _guard(svc().compare, da, na, db, nb)


@app.get("/lab/options/{modality}", tags=["lab"])
def lab_options(modality: str):
    if modality not in ("image", "audio", "video"):
        raise HTTPException(404, "unknown modality")
    return svc().lab_options(modality)


@app.post("/lab", tags=["lab"])
async def lab(file: UploadFile = File(...), kind: str = Form(...), strengths: str = Form("")):
    data, name = await _read(file)
    vals = [float(x) for x in strengths.split(",") if x.strip()] or None
    return _guard(svc().lab, data, name, kind, vals)


@app.post("/report", tags=["reports"])
async def report(file: UploadFile = File(...), case_id: str | None = Form(None), analyst: str = Form("")):
    data, name = await _read(file)
    return _guard(svc().report, data, name, case_id or None, analyst)


# ------------------------------------------------------------------ investigations
@app.post("/cases", tags=["cases"])
def create_case(payload: dict = Body(...)):
    return _guard(
        svc().create_case, payload.get("name", ""), payload.get("description", ""), payload.get("analyst", "")
    )


@app.get("/cases", tags=["cases"])
def list_cases():
    return svc().list_cases()


@app.get("/cases/{case_id}", tags=["cases"])
def get_case(case_id: str):
    return _guard(svc().get_case, case_id)


@app.patch("/cases/{case_id}/status", tags=["cases"])
def set_status(case_id: str, payload: dict = Body(...)):
    return _guard(svc().set_case_status, case_id, payload.get("status", ""))


@app.post("/cases/{case_id}/notes", tags=["cases"])
def add_note(case_id: str, payload: dict = Body(...)):
    return _guard(svc().add_case_note, case_id, payload.get("text", ""))


@app.get("/cases/{case_id}/report", tags=["cases"])
def case_report(case_id: str):
    return _guard(svc().case_report, case_id)


@app.delete("/cases/{case_id}", tags=["cases"])
def delete_case(case_id: str):
    svc().delete_case(case_id)
    return JSONResponse({"deleted": case_id})


# ------------------------------------------------------------------ benchmarks / MLOps
@app.get("/benchmarks", tags=["benchmarks"])
def benchmarks():
    return svc().benchmarks()


@app.get("/benchmarks/stats/{modality}", tags=["benchmarks"])
def feature_stats(modality: str):
    return svc().feature_stats(modality)


@app.get("/registry", tags=["mlops"])
def registry():
    return svc().registry_list()


@app.post("/registry/promote", tags=["mlops"])
def promote(payload: dict = Body(...)):
    return _guard(svc().promote, payload["modality"], payload["algorithm"], int(payload["version"]))


@app.get("/monitoring/{modality}", tags=["mlops"])
def monitoring(modality: str):
    return svc().monitoring(modality)


@app.get("/predictions", tags=["mlops"])
def predictions(limit: int = 200):
    return svc().prediction_log(limit)


@app.post("/feedback", tags=["mlops"])
def feedback(payload: dict = Body(...)):
    return _guard(svc().feedback, payload["record_id"], int(payload["label"]))


@app.get("/retrain/{modality}/check", tags=["mlops"])
def retrain_check(modality: str):
    return svc().retrain_check(modality)


@app.post("/retrain/{modality}", tags=["mlops"])
def retrain(modality: str, payload: dict = Body(default={})):
    return _guard(svc().retrain, modality, bool(payload.get("force", False)))


@app.get("/experiments", tags=["mlops"])
def experiments(limit: int = 200):
    return svc().experiments(limit)


@app.get("/dataset/versions", tags=["mlops"])
def dataset_versions():
    return svc().dataset_versions()
