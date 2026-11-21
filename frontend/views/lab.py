"""
ForensiX - Forensic Stress Laboratory
Empirical resilience testing across synthetic transformation vectors.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import ui
from forensix.utils import modality_from_extension


def render(backend):
    ui.hero(
        "Forensic Stress Laboratory",
        "Empirical resilience testing: simulate hostile or routine real-world transmission channels (JPEG re-compression, "
        "spatial down-sampling, Gaussian noise, blur, MP3 re-encoding) to observe classification curve stability.",
        tag="Stress Testing",
    )
    if not ui.need_models(backend):
        return
    name, data = ui.pick_file("Media file", "lab", ui.EXT)
    if not data:
        st.info("Analyze a file first (or upload one here).")
        return
    modality = modality_from_extension(name)
    opts = backend.lab_options(modality)

    st.markdown(
        """
        <div class="fx-card">
            <div class="fx-card-header">
                <span class="fx-card-title">Select Degradation Vector & Sweep Range</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    kind = st.radio("Degradation", list(opts), format_func=lambda k: opts[k]["label"], horizontal=True, label_visibility="collapsed")
    o = opts[kind]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        lo, hi = st.slider(
            "Degradation sweep range",
            float(o["min"]),
            float(o["max"]),
            (float(o["min"]), float(o["max"])),
            step=float(o["step"]),
        )
    with c2:
        n = st.slider("Sweep sample resolution (steps)", 3, 8, 6)
    strengths = [round(lo + (hi - lo) * i / (n - 1), 3) for i in range(n)]

    if st.button("Execute Degradation Sweep", type="primary", width="stretch"):
        with st.spinner("Applying synthetic transformations and executing multi-step inference..."):
            try:
                st.session_state["lab_res"] = (name, kind, backend.lab(data, name, kind, strengths))
            except Exception as e:
                st.error(str(e))
                return
    st.markdown("</div>", unsafe_allow_html=True)

    saved = st.session_state.get("lab_res")
    if not saved or saved[0] != name or saved[1] != kind:
        return
    res = saved[2]
    curve = res["curve"]
    ok = [c for c in curve if c["analysable"] and c["strength"] is not None]
    p0 = curve[0]["p_manipulated"]

    st.markdown(
        """
        <div class="fx-card">
            <div class="fx-card-header">
                <span class="fx-card-title">Resilience Trajectory Curve</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    # Shaded calibrated regions
    fig.add_hrect(y0=0.65, y1=1, fillcolor="rgba(194, 109, 102, 0.16)", opacity=1, line_width=0, annotation_text="Manipulated Zone", annotation_position="top left", annotation_font_color="rgba(236, 238, 242, 0.45)")
    fig.add_hrect(y0=0, y1=0.35, fillcolor="rgba(122, 154, 139, 0.16)", opacity=1, line_width=0, annotation_text="Authentic Zone", annotation_position="bottom left", annotation_font_color="rgba(236, 238, 242, 0.45)")
    fig.add_scatter(
        x=[c["strength"] for c in ok],
        y=[c["p_manipulated"] for c in ok],
        mode="lines+markers",
        line=dict(color="#7A9A8B", width=2.5),
        marker=dict(color="#8DB3A2", size=8, line=dict(color="#161920", width=1.5)),
        name="Degraded Specimen",
    )
    fig.add_hline(y=p0, line_dash="dash", line_color="rgba(255, 255, 255, 0.35)", annotation_text=f"Pristine Baseline ({p0:.2f})", annotation_font_color="#ECEEF2")
    fig.update_layout(
        height=320,
        yaxis=dict(range=[0, 1], title="P(manipulated)", gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
        xaxis=dict(title=res["label"], gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
        margin=dict(l=10, r=10, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
    )
    st.plotly_chart(fig, width="stretch")

    worst = max((abs(c["delta"]) for c in ok), default=0)
    flipped = any((c["p_manipulated"] >= 0.5) != (p0 >= 0.5) for c in ok)
    if flipped:
        st.error(
            "Classification Inversion Warning: The model flips decisions across this operational spectrum. "
            "Evidence processed through equivalent compression pipelines must be flagged as unreliable."
        )
    elif worst > 0.2:
        st.warning(
            f"Vulnerability Observed: Model confidence shifts by {100 * worst:.0f} percentage points across this sweep, "
            f"though categorical polarity is maintained."
        )
    else:
        st.success("Stable Robustness: Model confidence remains well-calibrated and resilient across this stress profile.")
    st.markdown("</div>", unsafe_allow_html=True)

    if any(not c["analysable"] for c in curve):
        st.caption("Extremes of degradation caused payload failure (unanalysable) and are excluded from the trajectory.")

    c_table, c_preview = st.columns([1.5, 1], gap="medium")
    with c_table:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Numerical Trajectory Table</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            [
                {
                    "Strength Vector": c["label"],
                    "P(manipulated)": None if c["p_manipulated"] is None else round(c["p_manipulated"], 3),
                    "Shift vs Baseline": None if c["delta"] is None else f"{100 * c['delta']:+.1f}%",
                }
                for c in curve
            ],
            width="stretch",
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c_preview:
        if res.get("preview"):
            st.markdown(
                """
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">Specimen Degradation Sample</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            ui.b64_image(
                res["preview"]["png_b64"], caption=f"Specimen at {res['label']} = {res['preview']['strength']}", width=320
            )
            st.markdown("</div>", unsafe_allow_html=True)
