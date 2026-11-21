"""
ForensiX - Benchmarks & Empirical Evaluation
Model comparisons, FDR statistical validation, robustness, and generative consistency.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import ui


def render(backend):
    ui.hero(
        "Forensic Benchmarks & Empirical Evaluation",
        "Empirical performance metrics across modalities, statistical feature validation, degradation stress-testing, "
        "and generative source consistency.",
        tag="Comparative Benchmarking",
    )
    b = backend.benchmarks()
    if not b["model_comparison"]:
        st.info("No benchmark results yet. Run the pipeline (`make demo` or `make pipeline`).")
        return
    t1, t2, t3, t4, t5 = st.tabs(
        ["Model Architecture Comparison", "Statistical Feature Validation", "Global SHAP Importance", "Channel Robustness", "Generator Consistency"]
    )
    with t1:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Test Split Metric Comparison Across Candidate Architectures</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        df = pd.DataFrame(b["model_comparison"])
        metric = st.radio("Evaluation Metric", ["f1", "accuracy", "precision", "recall", "roc_auc"], horizontal=True, label_visibility="collapsed")
        fig = px.bar(
            df,
            x="modality",
            y=metric,
            color="model",
            barmode="group",
            range_y=[0, 1.05],
            color_discrete_sequence=["#7A9A8B", "#8E8BA7", "#6B9099", "#C89B6A"],
        )
        fig.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
            margin=dict(t=20, b=20),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#ECEEF2", size=12)),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
            legend=dict(font=dict(color="#ECEEF2"), title=dict(font=dict(color="#959CA8"))),
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df.round(3), width="stretch", hide_index=True)
        st.caption(
            "Test-split evaluation scores. Optimal models per modality are selected exclusively on the validation split; "
            "the test split is untouched until final evaluation. `n_test` reflects the sample size behind each benchmark."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with t2:
        st.markdown("#### Hypothesis Testing & FDR Correction")
        for m, s in b["statistics"].items():
            st.markdown(
                f"""
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">{m.capitalize()} Modality Separation</span>
                    </div>
                    <div style="font-size:0.88rem; color:var(--text-secondary); margin-bottom:12px;">
                        <b>{s['n_significant_fdr05']}</b> of <b>{s['n_features_tested']}</b> features differ 
                        significantly between real and manipulated media (FDR Benjamini-Hochberg corrected, 5%); 
                        <b>{len(s['strong_effect_features'])}</b> demonstrate large effect size (|d| >= 0.8).
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(s["top_features"]), width="stretch", hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        if b["statistics"]:
            st.caption(next(iter(b["statistics"].values()))["method"])
    with t3:
        st.markdown("#### Global Feature Attribution Across Modalities")
        for m, s in b["shap"].items():
            st.markdown(
                f"""
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">{m.capitalize()} Baseline Model: {s['model']}</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.caption("Top Attributed Features")
                c1.dataframe(pd.Series(s["top_features"], name="Mean |SHAP|").to_frame(), width="stretch")
            with c2:
                st.caption("Indicator Family Totals")
                c2.dataframe(pd.Series(s["indicator_groups"], name="Mean |SHAP|").to_frame(), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
    with t4:
        rb = pd.DataFrame(b["robustness"])
        if rb.empty:
            st.info("Run `make robustness` to populate Channel Robustness telemetry.")
        else:
            mod = st.selectbox("Select Modality", sorted(rb["modality"].unique()))
            d = rb[rb["modality"] == mod]
            st.markdown(
                f"""
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">Transmission & Compression Robustness ({mod.upper()})</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            fig = px.bar(
                d,
                y="transformation",
                x="f1",
                orientation="h",
                color=d["transformation"].eq("Clean").map({True: "Clean Reference", False: "Degraded Channel"}),
                color_discrete_map={"Clean Reference": "#7A9A8B", "Degraded Channel": "#8E8BA7"},
                range_x=[0, 1.02],
            )
            fig.update_layout(
                height=60 + 34 * len(d),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
                margin=dict(t=20, b=20),
                legend_title="",
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", tickfont=dict(color="#ECEEF2")),
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                d[
                    [
                        "transformation",
                        "family",
                        "f1",
                        "clean_f1",
                        "f1_drop",
                        "recall_fake",
                        "recall_real",
                        "n_scored",
                        "n_failed",
                    ]
                ].round(3).rename(columns={
                    "transformation": "Transformation",
                    "family": "Category",
                    "f1": "F1 Score",
                    "clean_f1": "Clean F1",
                    "f1_drop": "Delta Drop",
                    "recall_fake": "Recall (Manipulated)",
                    "recall_real": "Recall (Authentic)",
                }),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Degraded copies are generated exclusively from the held-out TEST split. "
                "`n_failed` tallies samples rendered unanalysable post-transformation."
            )
            st.markdown("</div>", unsafe_allow_html=True)
    with t5:
        gn = pd.DataFrame(b["generators"])
        if gn.empty:
            st.info("Run `make generators` to compute generator-wise consistency telemetry.")
        else:
            mod = st.selectbox("Select Modality", sorted(gn["modality"].unique()), key="gen_mod")
            d = gn[gn["modality"] == mod].sort_values("detection_rate")
            st.markdown(
                f"""
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">Generative Synthesis Cross-Consistency ({mod.upper()})</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            fig = px.bar(
                d,
                y="source",
                x="detection_rate",
                orientation="h",
                range_x=[0, 1.02],
                color=d["low_support"].map({True: "Low Support (<20)", False: "Adequate Support"}),
                color_discrete_map={"Low Support (<20)": "#C89B6A", "Adequate Support": "#7A9A8B"},
                error_x=d["ci95_high"] - d["detection_rate"],
                error_x_minus=d["detection_rate"] - d["ci95_low"],
            )
            fig.update_layout(
                height=80 + 40 * len(d),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
                margin=dict(t=20, b=20),
                legend_title="",
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", tickfont=dict(color="#ECEEF2")),
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(d.round(3), width="stretch", hide_index=True)
            st.caption(
                "Generator labels are used to inspect blind-spots across synthetic architectures. "
                "Sources with small test sample pools exhibit wider confidence intervals and must not be over-extrapolated."
            )
            st.markdown("</div>", unsafe_allow_html=True)
