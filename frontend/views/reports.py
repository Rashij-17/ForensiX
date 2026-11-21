"""
ForensiX - Calibrated Forensic Reporting
Synthesize formal forensic assessment documents.
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

import ui


def render(backend):
    ui.hero(
        "Calibrated Forensic Reporting",
        "Synthesize comprehensive, legally defensive forensic assessment documents. "
        "Formulated with probabilistic framing, feature attribution, and analytical boundary disclosures.",
        tag="Document Synthesis",
    )
    if not ui.need_models(backend):
        return
    name, data = ui.pick_file("Media file", "rep", ui.EXT)
    if not data:
        st.info("Analyze a file first (or select/upload one here to generate a report).")
        return

    cases = backend.list_cases()
    options = {"Stand-alone report (no linked dossier)": None, **{f"{c['id']} — {c['name']}": c["id"] for c in cases}}

    st.markdown(
        """
        <div class="fx-card">
            <div class="fx-card-header">
                <span class="fx-card-title">Report Metadata & Attribution Parameters</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.5, 1], gap="medium")
    with c1:
        st.markdown("#### Dossier Attribution")
        case = c1.selectbox("Include case dossier details", list(options), label_visibility="collapsed")
    with c2:
        st.markdown("#### Examiner Signature")
        analyst = c2.text_input("Lead Analyst / Examiner identifier", label_visibility="collapsed", placeholder="e.g. Examiner #402")

    if st.button("Generate Forensic Assessment Report", type="primary", width="stretch"):
        with st.spinner("Synthesizing statistical context, model comparisons, and report layout..."):
            try:
                st.session_state["report"] = (name, backend.report(data, name, options[case], analyst))
            except Exception as e:
                st.error(str(e))
                return
    st.markdown("</div>", unsafe_allow_html=True)

    saved = st.session_state.get("report")
    if not saved or saved[0] != name:
        return
    rep = saved[1]

    st.markdown(
        """
        <div class="fx-card">
            <div class="fx-card-header">
                <span class="fx-card-title">Export & Document Preview</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    d1, d2, _ = st.columns([1, 1, 2.5])
    with d1:
        d1.download_button("Export Formal Report (HTML)", rep["html"], f"forensix_report_{name}.html", "text/html", width="stretch")
    with d2:
        d2.download_button("Export Dossier (Markdown)", rep["markdown"], f"forensix_report_{name}.md", "text/markdown", width="stretch")

    st.markdown('<div style="margin: 14px 0;"></div>', unsafe_allow_html=True)
    components.html(rep["html"], height=950, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)
