import re

import pytest
from fastapi.testclient import TestClient

from forensix.cases import CaseStore
from forensix.inference import LIMITATIONS, ForensiXEngine
from forensix.report import build_report, case_report


@pytest.fixture(scope="module")
def client(project, monkeypatch_module):
    monkeypatch_module.setenv("FORENSIX_HOME", str(project.root))
    from api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _post(client, path, f, **data):
    return client.post(path, files={"file": (f.name, f.read_bytes())}, data=data)


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and set(j["modalities"]) == {"image", "audio", "video"}


@pytest.mark.parametrize("name,mod", [("fake.png", "image"), ("fake.wav", "audio"), ("fake.mp4", "video")])
def test_analyze_each_modality_auto_detected(client, sample_files, name, mod):
    r = _post(client, "/analyze", sample_files / name)
    assert r.status_code == 200
    j = r.json()
    assert j["modality"] == mod and 0 <= j["probability_manipulated"] <= 1
    assert abs(j["probability_manipulated"] + j["probability_authentic"] - 1) < 1e-9
    assert j["verdict"]["label"] in {"manipulated", "authentic", "inconclusive"}
    assert len(j["explanation"]["contributions"]) == 8 and "media" not in j
    # SHAP additivity: base value + sum of SHAP values equals the reported model output
    ex = j["explanation"]
    assert abs(ex["base_value"] + sum(ex["all_shap"].values()) - ex["output_value"]) < 1e-9


def test_synthetic_fake_image_is_flagged_and_real_is_not(client, sample_files):
    assert _post(client, "/analyze", sample_files / "fake.png").json()["probability_manipulated"] > 0.5
    assert _post(client, "/analyze", sample_files / "real.jpg").json()["probability_manipulated"] < 0.5


def test_bad_inputs_return_clean_errors(client, tmp_path):
    t = tmp_path / "x.txt"
    t.write_text("hello")
    assert _post(client, "/analyze", t).status_code == 422
    c = tmp_path / "broken.png"
    c.write_bytes(b"garbage")
    assert _post(client, "/analyze", c).status_code == 422
    assert client.post("/analyze", files={"file": ("e.png", b"")}).status_code == 400
    assert client.get("/cases/DOES-NOT-EXIST").status_code == 404


def test_uploaded_media_is_not_retained(client, sample_files, project):
    before = {p.name for p in project.root.rglob("*") if p.is_file()}
    _post(client, "/analyze", sample_files / "fake.png")
    new = {p.name for p in project.root.rglob("*") if p.is_file()} - before
    assert not any(n.lower().endswith((".png", ".jpg", ".wav", ".mp4")) for n in new)


def test_evidence_compare_and_lab(client, sample_files):
    ev = _post(client, "/evidence", sample_files / "fake.png").json()
    assert {"frequency", "noise"} <= set(ev) and all(len(v) > 1000 for v in ev.values())
    r = client.post(
        "/compare",
        files={
            "file_a": ("a.png", (sample_files / "fake.png").read_bytes()),
            "file_b": ("b.jpg", (sample_files / "real.jpg").read_bytes()),
        },
    )
    j = r.json()
    assert r.status_code == 200 and 0 <= j["similarity"]["ssim"] <= 1 and "figure" in j
    r = client.post(
        "/compare",
        files={
            "file_a": ("a.png", (sample_files / "fake.png").read_bytes()),
            "file_b": ("b.wav", (sample_files / "fake.wav").read_bytes()),
        },
    )
    assert r.status_code == 422 and "same modality" in r.json()["detail"]
    lab = _post(client, "/lab", sample_files / "fake.png", kind="noise", strengths="2,10,20").json()
    assert lab["curve"][0]["label"] == "original" and len(lab["curve"]) == 4 and "preview" in lab
    assert _post(client, "/lab", sample_files / "fake.png", kind="bogus").status_code == 422


def test_case_workflow_end_to_end(client, sample_files):
    cid = client.post("/cases", json={"name": "Case A", "description": "demo", "analyst": "T"}).json()["id"]
    assert client.post("/cases", json={"name": "  "}).status_code == 422
    _post(client, "/analyze", sample_files / "fake.png", case_id=cid)
    _post(client, "/analyze", sample_files / "real.jpg", case_id=cid)
    case = client.get(f"/cases/{cid}").json()
    assert case["stats"]["n_analyses"] == 2 and case["stats"]["n_flagged"] >= 1
    assert client.patch(f"/cases/{cid}/status", json={"status": "closed"}).json()["status"] == "closed"
    assert client.patch(f"/cases/{cid}/status", json={"status": "bogus"}).status_code == 422
    rep = client.get(f"/cases/{cid}/report").json()
    assert "fake.png" in rep["markdown"] and "Limitations" in rep["markdown"]
    assert any(c["id"] == cid for c in client.get("/cases").json())
    client.delete(f"/cases/{cid}")
    assert client.get(f"/cases/{cid}").status_code == 404


def test_feedback_and_monitoring_endpoints(client, sample_files):
    rid = _post(client, "/analyze", sample_files / "fake.png").json()["record_id"]
    assert client.post("/feedback", json={"record_id": rid, "label": 1}).json()["updated"] is True
    assert client.post("/feedback", json={"record_id": "nope", "label": 1}).json()["updated"] is False
    assert client.post("/feedback", json={"record_id": rid, "label": 7}).status_code == 422
    assert client.get("/monitoring/image").json()["n_logged"] >= 1
    assert client.get("/benchmarks").json()["model_comparison"]
    assert len(client.get("/registry").json()) == 12


# ----------------------------------------------------------------------------------- language & report
@pytest.mark.parametrize(
    "p,label",
    [(0.95, "manipulated"), (0.5, "inconclusive"), (0.1, "authentic"), (0.65, "manipulated"), (0.35, "inconclusive")],
)
def test_verdict_bands(project, p, label):
    assert ForensiXEngine(project).verdict(p)["label"] == label


def test_report_uses_calibrated_language_and_states_limitations(project, sample_files):
    eng = ForensiXEngine(project)
    for name in ("fake.png", "real.jpg"):
        res = eng.analyze_bytes((sample_files / name).read_bytes(), name, log=False)
        res.pop("media")
        rep = build_report(res, project)
        md = rep["markdown"]
        for must in ("Authenticity assessment", "Model results", "Key forensic indicators", "Limitations"):
            assert must in md
        assert all(lim in md for lim in LIMITATIONS)
        # no un-negated claims of proof / certainty ("does not prove" is exactly the calibrated wording we want)
        assert not re.search(
            r"(?<!not )(?<!n't )\bproves?\b|\bproven\b|\bdefinitely\b|\bcertainly\b|100% (fake|real)|is a deepfake",
            md,
            re.I,
        )
        assert "not proof" in md.lower() or "not legal" in md.lower()
        assert res["sha256"] in md and "<html" in rep["html"] and "ForensiX Forensic Report" in rep["html"]


def test_report_html_escapes_untrusted_text(project, sample_files):
    eng = ForensiXEngine(project)
    res = eng.analyze_bytes((sample_files / "fake.png").read_bytes(), "<script>alert(1)</script>.png", log=False)
    res.pop("media")
    html = build_report(res, project)["html"]
    assert "<script>alert(1)</script>" not in html


def test_case_store_cascade_and_summary(project):
    s = CaseStore(project)
    c = s.create("X")
    assert s.get(c["id"])["stats"]["n_analyses"] == 0 and case_report(s.get(c["id"]), project)["markdown"]
    s.delete(c["id"])
    with pytest.raises(KeyError):
        s.get(c["id"])
