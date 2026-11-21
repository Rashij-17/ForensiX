"""
ForensiX - MLOps & Continuous Model Governance
Dataset lineage, experiment tracking, model registry, and drift monitoring.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui


def render(backend):
    ui.hero(
        "MLOps & Continuous Model Governance",
        "Continuous machine learning lifecycle controls: immutable dataset versioning, MLflow experiment lineage, "
        "model registry governance, operational drift monitoring, and gated candidate retraining triggers.",
        tag="Continuous MLOps",
    )
    t1, t2, t3, t4 = st.tabs(["Dataset Lineage", "Experiment Tracking", "Production Model Registry", "Operational Drift & Retraining Gate"])
    with t1:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Immutable Dataset Version Corpus</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        v = backend.dataset_versions()
        if v:
            rows = [
                {
                    "Version": x["version"],
                    "Committed": x["date"],
                    "Images": x["n_samples"]["image"],
                    "Audio": x["n_samples"]["audio"],
                    "Video": x["n_samples"]["video"],
                    "Total Corpus": x["n_total"],
                    "Preprocessing Hash": x["preprocessing_version"],
                    "Feature Extraction": x["feature_version"],
                    "Verified Additions": x["verified_production_samples_added"],
                    "Remarks": x["note"],
                }
                for x in v
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No dataset versions recorded yet.")
        st.caption(
            "An immutable version snapshot is committed whenever raw files, preprocessing pipelines, or verified feedback samples change. "
            "DVC (see `dvc.yaml`) tracks binary assets."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with t2:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">MLflow Experiment Telemetry</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        ex = backend.experiments()
        if ex:
            df = pd.DataFrame(ex)
            st.dataframe(df.round(3), width="stretch", hide_index=True)
        else:
            st.info("No MLflow runs found. Execute training pipeline (`make train`). MLflow UI available at port 5000 (`make mlflow`).")
        st.markdown("</div>", unsafe_allow_html=True)
    with t3:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Model Registry & Stage Governance</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        reg = backend.registry_list()
        if reg:
            df = pd.DataFrame(reg)
            df["test_f1"] = df["metrics"].map(lambda m: m.get("test_f1"))
            df["val_f1"] = df["metrics"].map(lambda m: m.get("val_f1"))
            show = df[["display", "modality", "stage", "val_f1", "test_f1", "dataset_version", "created", "notes"]].rename(
                columns={
                    "display": "Model Identifier",
                    "modality": "Modality",
                    "stage": "Stage",
                    "val_f1": "Val F1",
                    "test_f1": "Test F1",
                    "dataset_version": "Dataset",
                    "created": "Registered At",
                    "notes": "Governance Notes",
                }
            )
            st.dataframe(show.round(3), width="stretch", hide_index=True)

            st.markdown('<div style="margin: 20px 0; border-top: 1px solid var(--border-subtle);"></div>', unsafe_allow_html=True)
            st.markdown("#### Promote Candidate Model to Production")
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            cand = df[df["stage"] != "production"]
            choice = c1.selectbox(
                "Select Candidate Version", cand["display"].tolist() if len(cand) else ["No candidate models in registry"], label_visibility="collapsed"
            )
            if c2.button("Approve & Promote", type="primary") and len(cand) and choice:
                r = cand[cand["display"] == choice].iloc[0]
                backend.promote(r["modality"], r["algorithm"], int(r["version"]))
                st.success(f"{choice} promoted to PRODUCTION state for {r['modality'].upper()}.")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with t4:
        st.markdown("#### Live Telemetry & Feature Drift Monitoring")
        mod = st.radio("Select Target Modality", ["image", "audio", "video"], horizontal=True, label_visibility="collapsed")
        mon = backend.monitoring(mod)

        # KPI row
        a, b, c = st.columns(3, gap="medium")
        a.metric("Production Inferences", mon["n_logged"])
        b.metric("Drift Readiness", "Evaluated" if mon["drift_judged"] else f"Needs {mon['min_samples']}+")
        c.metric(
            "Features Drifted",
            f"{len(mon['drifted_features'])} ({mon['drift_share']:.0%})" if mon["drift_judged"] else "Awaiting baseline",
        )

        if mon["feature_psi"]:
            st.markdown(
                """
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">Population Stability Index (PSI vs. Training Baseline)</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            s = pd.Series(mon["feature_psi"]).sort_values(ascending=False).head(12).iloc[::-1]
            fig = go.Figure(
                go.Bar(
                    x=s.values,
                    y=s.index,
                    orientation="h",
                    marker_color=["#C26D66" if x > mon["psi_threshold"] else "#7A9A8B" for x in s.values],
                    marker_line=dict(width=0),
                )
            )
            fig.add_vline(x=mon["psi_threshold"], line_dash="dash", line_color="#C89B6A", annotation_text=f"PSI Threshold ({mon['psi_threshold']})", annotation_font_color="#E2B685")
            fig.update_layout(
                height=60 + 30 * len(s),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
                margin=dict(t=20, b=20),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)", tickfont=dict(color="#ECEEF2")),
            )
            st.plotly_chart(fig, width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        ps = mon.get("prediction_stats") or {}
        if ps:
            st.markdown(
                """
                <div class="fx-card">
                    <div class="fx-card-header">
                        <span class="fx-card-title">Operational Prediction Histogram Distribution</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            fig.add_bar(
                x=[f"{i / 10:.1f}" for i in range(10)],
                y=ps["prob_hist"],
                name="Live Telemetry",
                marker_color="#8E8BA7",
                marker_line=dict(width=0),
            )
            if ps.get("baseline_prob_hist"):
                fig.add_scatter(
                    x=[f"{i / 10:.1f}" for i in range(10)],
                    y=ps["baseline_prob_hist"],
                    name="Validation Baseline (counts)",
                    mode="lines+markers",
                    line=dict(color="#7A9A8B", width=2),
                    marker=dict(color="#8DB3A2", size=6),
                )
            fig.update_layout(
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
                margin=dict(t=20, b=20),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.06)", tickfont=dict(color="#959CA8")),
                legend=dict(font=dict(color="#ECEEF2")),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"Share flagged: {ps['share_flagged']:.0%} · Mean confidence: {ps['conf_mean']:.2f}"
                + (f" (Baseline {ps['baseline_conf_mean']:.2f})" if ps.get("baseline_conf_mean") is not None else "")
            )
            st.markdown("</div>", unsafe_allow_html=True)

        perf = mon.get("performance") or {}
        if perf:
            st.markdown(
                f"**Production Performance on Verified Labels** (n={perf['n_labeled']}"
                f"{'' if perf['judged'] else ', insufficient support'}): F1 **{perf['f1']:.3f}** · Precision **{perf['precision']:.3f}** · Recall **{perf['recall']:.3f}**"
            )
        else:
            st.caption("No ground-truth verifications recorded. Use *Submit ground truth verification* on the Analyze view to log verified outcomes.")

        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Automated Retraining Gating Trigger</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        dec = backend.retrain_check(mod)
        if dec["retrain"]:
            st.error("Retraining Trigger Tripped: " + "; ".join(dec["reasons"]))
        else:
            st.success("Stable Operations: No critical feature drift or performance degradation detected. Retraining trigger idle.")
        c_force, c_btn = st.columns([2, 1], vertical_alignment="center")
        force = c_force.checkbox("Force retraining cycle bypass (override trigger check)")
        if c_btn.button(f"Execute Retraining ({mod.upper()})", type="primary", width="stretch"):
            with st.spinner("Retraining candidate architectures and evaluating validation gates..."):
                out = backend.retrain(mod, force)
            if out["action"] == "skipped":
                st.info(out["reason"])
            elif out["action"] == "promoted":
                st.success(
                    f"Candidate Passed Gate & Promoted: Test F1 {out['candidate_test_f1']:.3f} outperformed production ({out['production_test_f1']:.3f})."
                )
            else:
                st.warning(
                    f"Candidate Rejected at Gate: Test F1 {out['candidate_test_f1']:.3f} failed to beat current production ({out['production_test_f1']:.3f})."
                )
        st.caption(
            "Gate Rule: A retrained candidate model is promoted to production strictly if its F1 score on the untouched test split exceeds the existing production benchmark."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        log = backend.prediction_log(100)
        if log:
            with st.expander("Telemetry Audit Log (Recent 100 Inferences)"):
                st.dataframe(
                    pd.DataFrame(log)[["ts", "modality", "filename", "probability", "model", "verified_label", "id"]].rename(
                        columns={
                            "ts": "Timestamp",
                            "modality": "Modality",
                            "filename": "Filename",
                            "probability": "P(manipulated)",
                            "model": "Model",
                            "verified_label": "Ground Truth",
                            "id": "Telemetry ID",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
