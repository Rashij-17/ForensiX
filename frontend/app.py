"""
ForensiX - Multimodal Media Forensics
Deep Muted Slate Architecture with Layered Surfaces
Run: streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import ui
from backend import get_backend
from views import (
    about,
    analyze,
    benchmarks,
    compare,
    evidence,
    investigations,
    lab,
    mlops,
    reports,
)

# --------------------------------------------------
# PAGE CONFIGURATION & ROOT CSS
# --------------------------------------------------

ui.setup_page()

st.markdown("""
<style>
/* ================= TOP NAVBAR ALIGNMENT & BRAND ================= */
.top-brand {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text-primary);
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin: 0;
    padding-right: 18px;
    white-space: nowrap;
}

.top-brand span {
    color: var(--accent-sage);
}

.top-brand-badge {
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 750;
    color: var(--accent-sage);
    background: rgba(122, 154, 139, 0.18);
    border: 1px solid rgba(122, 154, 139, 0.35);
    border-radius: var(--radius-pill);
    padding: 2px 8px;
}

.top-status-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-secondary);
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill);
    padding: 6px 14px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    white-space: nowrap;
}

.top-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent-sage);
    animation: pulseDot 2s infinite;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# ROUTER & BACKEND INITIALIZATION
# --------------------------------------------------

PAGES = {
    "Analyze": analyze,
    "Evidence": evidence,
    "Compare": compare,
    "Forensic Lab": lab,
    "Investigations": investigations,
    "Reports": reports,
    "Benchmarks": benchmarks,
    "MLOps": mlops,
    "About": about,
}


@st.cache_resource(show_spinner="Initializing ForensiX backend...")
def _backend():
    return get_backend()


backend = _backend()
status = backend.status()


# --------------------------------------------------
# TOP NAVIGATION HEADER (SPACIOUS & ELEGANT)
# --------------------------------------------------

col_brand, col_nav, col_status = st.columns([1.5, 7.0, 1.5], vertical_alignment="center")

with col_brand:
    st.markdown(
        """
        <div class="top-brand">
            Forensi<span>X</span>
            <span class="top-brand-badge">PRO</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav:
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Navigation",
            list(PAGES.keys()),
            default="Analyze",
            required=True,
            label_visibility="collapsed",
            key="top_nav_control",
        )
        page = selected if selected else "Analyze"
    else:
        page = st.radio(
            "Navigation",
            list(PAGES.keys()),
            horizontal=True,
            label_visibility="collapsed",
            key="top_nav_radio",
        )

with col_status:
    st.markdown(
        f"""
        <div style="text-align: right;">
            <div class="top-status-chip">
                <span class="top-status-dot"></span>
                <span>{backend.mode.upper()} · LIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-bottom: 0.9rem;"></div>', unsafe_allow_html=True)


# --------------------------------------------------
# COLLAPSIBLE SIDEBAR: TELEMETRY & PIPELINES
# --------------------------------------------------

with st.sidebar:
    st.markdown("### Production Pipelines")
    st.caption("Active statistical models evaluated per modality:")
    for model_name, model_status in status["production"].items():
        color = "#7A9A8B" if model_status else "#C89B6A"
        label = model_status if model_status else "PENDING"
        st.markdown(
            f"<div style='font-size:0.84rem; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;'>"
            f"<span style='color:var(--text-secondary);'>{model_name.capitalize()}</span>"
            f"<span style='color:{color}; font-weight:600; font-size:0.78rem; background:{color}22; padding:2px 8px; border-radius:9999px;'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Decision Protocol")
    st.caption(
        "Outputs are probabilistic estimates for investigative support, not legal proof."
    )


# --------------------------------------------------
# DEMO MODE NOTICE
# --------------------------------------------------

if status.get("demo_data"):
    st.markdown(
        """
        <div class="fx-warn">
            <b>Demo Mode Active:</b> Evaluated against synthetic verification data.
            Real-world forensic assessments require full benchmark ingestion (<code>make dataset</code>).
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# RENDER ACTIVE VIEW
# --------------------------------------------------

PAGES[page].render(backend)
