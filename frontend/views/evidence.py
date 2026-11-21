"""
ForensiX - Evidence Intelligence
Statistical hypothesis tests, signal & spectral analysis, and SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui


def render(backend):
    ui.hero(
        "Evidence Intelligence",
        "Empirical forensic measurements: comparing this file's extracted statistical distribution "
        "against authentic and manipulated reference corpuses.",
        tag="Signal Intelligence",
    )
    if not ui.need_models(backend):
        return
    name, data = ui.pick_file("Media file", "ev", ui.EXT)
    if not data:
        st.info("Analyze a file first (or upload one here).")
        return
    cur = ui.current()
    res = cur["result"] if cur and cur["name"] == name and cur["result"] and cur["result"].get("explanation") else None
    if res is None:
        with st.spinner("Analyzing media stream..."):
            res = backend.analyze(data, name, None, True, "")
            ui.set_current(name, data, res)

    t1, t2, t3 = st.tabs(["Statistical Hypothesis Tests", "Signal & Spectral Visualizations", "SHAP Feature Attribution"])
    with t1:
        stats_tab(backend, res)
    with t2:
        with st.spinner("Rendering spectral visualizations..."):
            figs = backend.evidence(data, name)
        titles = {
            "frequency": "Frequency Domain: 2D FFT & DCT Analysis",
            "noise": "High-Frequency Noise Residual Map",
            "waveform": "Temporal Audio Waveform Envelope",
            "spectral": "Spectral Log-Mel Spectrogram & MFCC Coefficients",
            "frames": "Sampled Video Frame Stream",
            "temporal": "Temporal Inter-Frame Warp Error & Anomalies",
        }
        notes = {
            "frequency": "Isolated spikes away from DC center in the 2D FFT indicate generative periodic upsampling grid artifacts. Organic captures demonstrate radially smooth energy falloff.",
            "spectral": "Synthetic vocoders often generate unnaturally smooth high frequencies or steep cutoff boundaries beyond 8-12 kHz.",
            "temporal": "Anomalies marked above 2 standard deviations denote unconstrained temporal instability or motion-compensation failure.",
        }
        for k, b in figs.items():
            st.markdown(
                f"""
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">{titles.get(k, k)}</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            ui.b64_image(b, width="stretch")
            if k in notes:
                st.caption(notes[k])
            st.markdown("</div>", unsafe_allow_html=True)
    with t3:
        ex = res["explanation"]
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            st.markdown(
                """
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">Individual Feature Attribution (SHAP)</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(ui.shap_chart(ex["contributions"]), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(
                """
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">Group Attribution Totals</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            gt = ex["group_totals"]
            fig = go.Figure(
                go.Bar(
                    x=list(gt.values()),
                    y=list(gt.keys()),
                    orientation="h",
                    marker_color=["#C26D66" if v > 0 else "#7A9A8B" for v in gt.values()],
                    marker_line=dict(width=0),
                )
            )
            fig.update_layout(
                title=dict(text="Net Attribution by Feature Group", font=dict(color="#ECEEF2", size=13, weight=600)),
                height=300,
                margin=dict(l=10, r=10, t=35, b=25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", zerolinecolor="rgba(255, 255, 255, 0.15)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)"),
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Attribution Coefficients Table</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(ex["contributions"])[["feature", "group", "description", "value", "shap", "direction"]]
        st.dataframe(df, width="stretch", hide_index=True)
        st.caption(
            f"Base value {ex['base_value']:.3f} + sum of SHAP values = model output {ex['output_value']:.3f} "
            f"(method: {ex['method']}). Positive values indicate divergence towards manipulation."
        )
        st.markdown("</div>", unsafe_allow_html=True)


def stats_tab(backend, res: dict):
    rows = backend.feature_stats(res["modality"])
    if not rows:
        st.info("Dataset-level statistics are not available yet. Run `make stats`.")
        return
    st.markdown(
        "Empirical hypothesis test distributions comparing **this file** against authentic and manipulated baselines. "
        "Evaluated via Mann-Whitney U tests with False Discovery Rate (FDR 5%) correction."
    )
    df = pd.DataFrame(rows)
    df["this_file"] = df["feature"].map(res["features"])
    df["z_vs_real"] = (df["this_file"] - df["mean_real"]) / (df["std_real"] + 1e-12)
    df["z_vs_fake"] = (df["this_file"] - df["mean_fake"]) / (df["std_fake"] + 1e-12)
    df["closer_to"] = np.where(df["z_vs_real"].abs() < df["z_vs_fake"].abs(), "authentic", "manipulated")
    top = df.sort_values("abs_d", ascending=False).head(15)

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Features Evaluated", len(df))
    c2.metric("Significant Separation (FDR 5%)", int(df["significant"].sum()))
    c3.metric("Closer to Manipulated Baseline", f"{int((top['closer_to'] == 'manipulated').sum())} / 15")

    st.markdown(
        """
        <div class="fx-card">
            <div class="fx-card-header">
                <span class="fx-card-title">Top Discriminative Features by Effect Size</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    show = top[
        ["feature", "group", "this_file", "mean_real", "mean_fake", "cohens_d", "mannwhitney_q", "closer_to"]
    ].rename(
        columns={
            "this_file": "This File",
            "mean_real": "Authentic μ",
            "mean_fake": "Manipulated μ",
            "cohens_d": "Cohen's d",
            "mannwhitney_q": "q-value (MWU)",
            "closer_to": "Closer To",
        }
    )
    st.dataframe(show, width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="fx-card">
            <div class="fx-card-header">
                <span class="fx-card-title">Empirical Distribution Inspection</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    feat = st.selectbox("Inspect distribution curve for feature", top["feature"].tolist())
    r = df[df["feature"] == feat].iloc[0]
    xs = np.linspace(
        min(r.mean_real - 3 * r.std_real, r.mean_fake - 3 * r.std_fake, r.this_file),
        max(r.mean_real + 3 * r.std_real, r.mean_fake + 3 * r.std_fake, r.this_file),
        200,
    )
    g = lambda m, s: np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))  # noqa: E731
    fig = go.Figure()
    fig.add_scatter(
        x=xs,
        y=g(r.mean_real, r.std_real + 1e-12),
        name="Authentic Baseline",
        line_color="#7A9A8B",
        fill="tozeroy",
        fillcolor="rgba(122, 154, 139, 0.2)",
    )
    fig.add_scatter(
        x=xs,
        y=g(r.mean_fake, r.std_fake + 1e-12),
        name="Manipulated Baseline",
        line_color="#C26D66",
        fill="tozeroy",
        fillcolor="rgba(194, 109, 102, 0.2)",
    )
    fig.add_vline(x=r.this_file, line_dash="dash", line_color="#ECEEF2", annotation_text="This File", annotation_font_color="#ECEEF2")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=35, b=30),
        title=dict(text=f"{feat}: {r['description']}", font=dict(color="#ECEEF2", size=13, weight=600)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", tickfont=dict(color="#959CA8")),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Welch 95% CI of mean delta: [{r.ci95_low:.4g}, {r.ci95_high:.4g}] · "
        f"Outlier share authentic: {r.outlier_share_real:.1%} / manipulated: {r.outlier_share_fake:.1%}."
    )
    st.markdown("</div>", unsafe_allow_html=True)
