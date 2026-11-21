"""
ForensiX - Analyze Workspace
Calm, layered minimalist design with 8px rhythm.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

import ui

FRONTEND_DIR = Path(__file__).resolve().parents[1]
HERO_IMAGE = FRONTEND_DIR / "assets" / "forensix_3d.png"


def render_hero(backend):
    left, right = st.columns([1.85, 1.15], vertical_alignment="center")
    with left:
        st.markdown(
            textwrap.dedent(f"""
                <div class="fx-hero">
                    <div class="fx-hero-tag">AUTHENTICITY INTELLIGENCE · DEEPSAFE CORPUS</div>
                    <div class="fx-hero-title">Multimodal Media Forensics</div>
                    <p class="fx-hero-desc">
                        Rigorous, probabilistic forensic screening across image, audio, and video streams. 
                        Extracts 114+ statistical, spatial, and spectral indicators with fully transparent 
                        SHAP attribution and empirical hypothesis testing.
                    </p>
                    <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:20px;">
                        <span class="fx-stat-pill">Engine: <b>{backend.mode.upper()}</b></span>
                        <span class="fx-stat-pill">Corpus: <b>6,184</b> specimens</span>
                        <span class="fx-stat-pill">Hypothesis Gate: <b>FDR 5%</b></span>
                        <span class="fx-stat-pill">Explainability: <b>SHAP</b></span>
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )
    with right:
        if HERO_IMAGE.exists():
            st.image(str(HERO_IMAGE), width="stretch")


def render_upload(backend):
    cases = backend.list_cases()
    options = {
        "Stand-alone Analysis (No Case Link)": None,
        **{f"{c['id']} — {c['name']}": c["id"] for c in cases if c["status"] != "closed"},
    }

    c_left, c_right = st.columns([1.8, 1.2], gap="medium")

    with c_left:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Source Media Intake</span>
                    <span class="fx-card-chip">RAM-Only Processing</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        up = st.file_uploader(
            "Upload media file",
            type=ui.EXT,
            help="Images, audio, or video files. Evaluated in-memory and never permanently stored.",
            label_visibility="collapsed",
            key="forensix_media_upload",
        )
        st.markdown(
            """
                <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
                    <span style="font-size:0.75rem; color:var(--text-secondary); background:var(--surface-subtle); padding:3px 10px; border-radius:9999px;">Image: JPG, PNG, WEBP</span>
                    <span style="font-size:0.75rem; color:var(--text-secondary); background:var(--surface-subtle); padding:3px 10px; border-radius:9999px;">Audio: WAV, MP3, FLAC</span>
                    <span style="font-size:0.75rem; color:var(--text-secondary); background:var(--surface-subtle); padding:3px 10px; border-radius:9999px;">Video: MP4, AVI, MOV</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_right:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Investigation Dossier</span>
                    <span class="fx-card-chip">Context</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        case_label = st.selectbox(
            "Attach to investigation dossier",
            list(options.keys()),
            label_visibility="collapsed",
        )
        explain = st.checkbox("Generate SHAP feature attribution", value=True)
        st.caption("SHAP quantifies exact mathematical feature attribution influencing model divergence.")
        st.markdown("</div>", unsafe_allow_html=True)

    return up, options, case_label, explain


def execute_analysis(backend, up, options, case_label, explain):
    data = up.getvalue()
    if st.button("Run Forensic Analysis", type="primary", width="stretch"):
        with st.spinner("Extracting forensic signatures and evaluating models..."):
            try:
                res = backend.analyze(data, up.name, options[case_label], explain, "")
                ui.set_current(up.name, data, res)
                return res
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return None
    return None


def show(res: dict, data: bytes, backend):
    v = res["verdict"]

    st.markdown('<div style="margin: 24px 0 16px 0;"></div>', unsafe_allow_html=True)
    st.markdown("### Authenticity Evaluation Results")

    # Row 1: Specimen Inspection & Gauge Verdict
    row1_left, row1_right = st.columns([1.1, 1.3], gap="medium")

    with row1_left:
        st.markdown(
            f"""
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Specimen Inspection</span>
                    <span class="fx-card-chip">{res['modality'].upper()}</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        ui.media_preview(data, res["filename"], res["modality"])
        st.markdown(
            f"""
                <div style="margin-top:14px; border-top: 1px solid var(--border-subtle); padding-top:10px; display:flex; justify-content:space-between; font-size:0.85rem;">
                    <div><span style="color:var(--text-secondary);">File:</span> <b>{res['filename']}</b></div>
                    <div><span style="color:var(--text-secondary);">Latency:</span> <b>{res.get('elapsed_s', '?')}s</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with row1_right:
        st.markdown(
            f"""
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Calibrated Verdict</span>
                    <span class="fx-card-chip">{v['evidence_strength'].upper()} CONFIDENCE</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div>{ui.badge(v["label"], v["text"])}</div>
                    <div style="font-size:0.82rem; color:var(--text-secondary);">
                        Model: <b style="color:var(--text-primary);">{res['model']['display']}</b>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            ui.gauge(res["probability_manipulated"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            f"Algorithm: **{res['model']['algorithm'].upper()}** · "
            f"Calibrated probability: **{res['probability_manipulated']:.1%}** manipulated vs. **{res['probability_authentic']:.1%}** authentic."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: 3-Column Info Cards
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Model Consensus</span>
                    <span class="fx-card-chip">Ensemble</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if res.get("model_results"):
            mr = pd.DataFrame(res["model_results"]).copy()
            mr["P(manipulated)"] = mr["probability_manipulated"].map(ui.pct)
            mr["role"] = mr["production"].map({True: "Production", False: "Candidate"})
            st.dataframe(mr[["model", "P(manipulated)", "role"]], width="stretch", hide_index=True)
            agree = len({p >= 0.5 for p in mr["probability_manipulated"]}) == 1
            if agree:
                st.success("All models agree on classification direction.")
            else:
                st.warning("Candidate algorithms diverge. Interpret cautiously.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"""
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Cryptographic Hash</span>
                    <span class="fx-card-chip">SHA-256</span>
                </div>
                <div style="font-size:0.8rem; color:var(--text-secondary); margin-bottom:6px;">Deterministic Ingestion Hash:</div>
                <code style="word-break: break-all; display:block; padding:8px 10px;">
                    {res['sha256']}
                </code>
                <div style="margin-top:12px; font-size:0.78rem; color:var(--text-secondary);">
                    Payload integrity sealed at ingestion time.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Ground Truth Feedback</span>
                    <span class="fx-card-chip">Telemetry</span>
                </div>
                <div style="font-size:0.82rem; color:var(--text-secondary); margin-bottom:12px;">
                    Record verified provenance to monitor operational drift:
                </div>
            """,
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Authentic", key="btn_v0", width="stretch"):
                out = backend.feedback(res["record_id"], 0)
                if out["updated"]:
                    st.success("Logged as Authentic.")
        with b2:
            if st.button("Manipulated", key="btn_v1", width="stretch"):
                out = backend.feedback(res["record_id"], 1)
                if out["updated"]:
                    st.success("Logged as Manipulated.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: SHAP Feature Attribution
    if res.get("explanation"):
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Explainable Attribution Matrix (SHAP)</span>
                    <span class="fx-card-chip">Interpretability</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            ui.shap_chart(res["explanation"]["contributions"]),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption("Right-extending bars indicate forensic features driving the score toward synthetic manipulation.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Calibrated Boundary Disclosures
    st.markdown(
        """
        <div class="fx-warn">
            <b>Calibrated Decision Support Principle:</b>
            Statistical models produce probabilities relative to reference data distributions. 
            This output is not definitive legal or judicial proof of forgery or authenticity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Analytical boundaries & system limitations"):
        for lim in res["limitations"]:
            st.markdown(f"- {lim}")


def render(backend):
    render_hero(backend)

    if not ui.need_models(backend):
        return

    up, options, case_label, explain = render_upload(backend)

    if up is not None:
        st.markdown(
            textwrap.dedent(f"""
                <div class="fx-card-subtle" style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:var(--text-secondary); font-size:0.8rem; text-transform:uppercase; font-weight:600;">Ingested:</span> 
                        <b style="color:var(--text-primary); font-size:0.95rem; margin-left:6px;">{up.name}</b>
                    </div>
                    <span style="font-size:0.8rem; color:var(--accent-sage); font-weight:600;">
                        {up.size / (1024 * 1024):.2f} MB
                    </span>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )

        execute_analysis(backend, up, options, case_label, explain)
        cur = ui.current()

        if cur and cur["name"] == up.name and cur["result"]:
            show(cur["result"], up.getvalue(), backend)
        else:
            st.info("File ingested. Click 'Run Forensic Analysis' above to commence.")
    else:
        st.markdown(
            textwrap.dedent("""
                <div class="fx-card" style="text-align:center; padding:36px 24px;">
                    <div style="font-size:1.1rem; font-weight:600; color:var(--text-primary); margin-bottom:4px;">No Media Ingested</div>
                    <div style="font-size:0.88rem; color:var(--text-secondary);">
                        Upload an image, audio clip or video above to begin evaluation. Test specimens are available in <code>test_samples/</code>.
                    </div>
                </div>
            """).strip(),
            unsafe_allow_html=True,
        )
