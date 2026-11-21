"""
ForensiX - Comparative Forensic Analysis
Differential forensic analysis between two versions of the same media specimen.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import ui


def render(backend):
    ui.hero(
        "Comparative Forensic Analysis",
        "Perform differential forensic analysis between two versions of the same media specimen (e.g. pristine capture vs. "
        "re-encoded/manipulated variant) to measure structural divergence and classification stability.",
        tag="Differential Forensics",
    )
    if not ui.need_models(backend):
        return

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Baseline Specimen (A)</span>
                    <span class="fx-card-chip">Original</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        a = st.file_uploader("Upload reference file (e.g. original)", type=ui.EXT, key="cmp_a", label_visibility="collapsed")
        st.caption("Pristine or reference baseline specimen.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Suspect Specimen (B)</span>
                    <span class="fx-card-chip">Suspect</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        b = st.file_uploader("Upload suspect file (e.g. re-encoded)", type=ui.EXT, key="cmp_b", label_visibility="collapsed")
        st.caption("Suspect, edited, or re-compressed specimen.")
        st.markdown("</div>", unsafe_allow_html=True)

    if not (a and b):
        st.info("Ingest two files of matching modalities to initiate comparative analysis.")
        return

    if st.button("Compute Differential Analysis", type="primary"):
        with st.spinner("Computing structural similarity, feature divergence, and scoring differentials..."):
            try:
                st.session_state["cmp_res"] = (
                    a.name,
                    b.name,
                    backend.compare(a.getvalue(), a.name, b.getvalue(), b.name),
                )
            except Exception as e:
                st.error(str(e))
                return

    saved = st.session_state.get("cmp_res")
    if not saved or saved[0] != a.name or saved[1] != b.name:
        return
    res = saved[2]
    s = res["scores"]

    st.markdown('<div style="margin: 20px 0 12px 0;"></div>', unsafe_allow_html=True)

    # Differential KPI Cards
    m1, m2, m3 = st.columns(3, gap="medium")
    m1.metric("Specimen A : P(manipulated)", ui.pct(s["p_a"]))
    m2.metric("Specimen B : P(manipulated)", ui.pct(s["p_b"]), delta=f"{100 * s['delta']:+.1f} pts", delta_color="inverse")
    m3.metric("Evaluated Modality", res["modality"].upper())

    # Divergence Visuals
    st.markdown(
        """
        <div class="fx-card">
            <div class="fx-card-header">
                <span class="fx-card-title">Structural Divergence Visualization</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    ui.b64_image(res["figure"], width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    # Metrics & Shifts
    c_left, c_right = st.columns([1, 1.25], gap="medium")
    with c_left:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Structural Similarity Index</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        sim = res["similarity"]
        st.dataframe(pd.DataFrame([{"Metric": k, "Value": round(v, 4) if isinstance(v, float) else v} for k, v in sim.items()]), hide_index=True, width="stretch")
        if res["modality"] == "image":
            st.caption("SSIM 1.0 represents structural equivalence; PSNR above 40 dB indicates imperceptible pixel divergence.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_right:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Largest Feature Shifts</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(s["largest_feature_shifts"]), hide_index=True, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    if abs(s["delta"]) > 0.25:
        st.markdown(
            """
            <div class="fx-warn">
                <b>High Compression / Transmission Sensitivity:</b>
                The divergence between versions exceeds 25 percentage points. The model's classification is sensitive to re-encoding artifacts.
            </div>
            """,
            unsafe_allow_html=True,
        )
