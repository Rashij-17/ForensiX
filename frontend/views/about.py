"""
ForensiX - Architecture & Governance Protocol
Ethical commitments, operational boundaries, and calibrated decision tiers.
"""

from __future__ import annotations

import streamlit as st

import ui


def render(backend):
    ui.hero(
        "Forensic Architecture & Governance Protocol",
        "Multimodal digital media forensics and authenticity intelligence: calibrated statistical indicators, "
        "explainable feature attribution, and empirical MLOps lifecycle governance.",
        tag="Governance & Ethics",
    )

    c1, c2 = st.columns([1.6, 1], gap="medium")

    with c1:
        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Core Operational Protocol</span>
                </div>
                <div style="color:var(--text-secondary); font-size:0.95rem; line-height:1.7;">
                    ForensiX extracts spatial, frequency, spectral, and temporal forensic signatures across 
                    <b>image, audio, and video</b> modalities. Every prediction is evaluated with transparent 
                    SHAP attribution, tested for channel degradation resilience, and communicated exclusively 
                    in calibrated probabilistic terms.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Ethical Commitments & Operational Bounds</span>
                </div>
                <div style="display:flex; flex-direction:column; gap:14px;">
                    <div style="display:flex; gap:14px; align-items:flex-start;">
                        <span style="font-size:1.2rem;">🛡️</span>
                        <div>
                            <b style="color:var(--text-primary); font-size:0.92rem;">Defensive Purpose Limitation</b>
                            <div style="font-size:0.85rem; color:var(--text-secondary);">Strictly designed as an authenticity verification tool. No generative or evasion features are provided.</div>
                        </div>
                    </div>
                    <div style="display:flex; gap:14px; align-items:flex-start;">
                        <span style="font-size:1.2rem;">⚖️</span>
                        <div>
                            <b style="color:var(--text-primary); font-size:0.92rem;">Calibrated Probabilistic Output</b>
                            <div style="font-size:0.85rem; color:var(--text-secondary);">Evaluations express statistical probabilities, never absolute proof or legal certainty of forgery.</div>
                        </div>
                    </div>
                    <div style="display:flex; gap:14px; align-items:flex-start;">
                        <span style="font-size:1.2rem;">🔒</span>
                        <div>
                            <b style="color:var(--text-primary); font-size:0.92rem;">Zero-Persistence RAM Privacy</b>
                            <div style="font-size:0.85rem; color:var(--text-secondary);">Media buffers are evaluated in-memory and scrubbed immediately post-inference. Only mathematical signatures are retained.</div>
                        </div>
                    </div>
                    <div style="display:flex; gap:14px; align-items:flex-start;">
                        <span style="font-size:1.2rem;">👤</span>
                        <div>
                            <b style="color:var(--text-primary); font-size:0.92rem;">Human-in-the-Loop Oversight</b>
                            <div style="font-size:0.85rem; color:var(--text-secondary);">Decision support architecture requiring qualified human analyst review for judicial or journalistic conclusions.</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="fx-card" style="border-top:3px solid var(--accent-sage);">
                <div class="fx-card-header">
                    <span class="fx-card-title">Calibrated Decision Thresholds</span>
                </div>
                <div class="fx-card-subtle" style="margin-bottom:10px; border-left:3px solid #7A9A8B;">
                    <div style="font-size:0.82rem; font-weight:700; color:#5D7D6E;">0% — 35% : AUTHENTIC TIER</div>
                    <div style="font-size:0.78rem; color:var(--text-secondary);">No strong indication of synthetic manipulation. Measurements align with genuine sensor baselines.</div>
                </div>
                <div class="fx-card-subtle" style="margin-bottom:10px; border-left:3px solid #C89B6A;">
                    <div style="font-size:0.82rem; font-weight:700; color:#9C6A2E;">35% — 65% : INCONCLUSIVE TIER</div>
                    <div style="font-size:0.78rem; color:var(--text-secondary);">Ambiguous indicator distribution. Requires human review, provenance audit, or higher-quality specimens.</div>
                </div>
                <div class="fx-card-subtle" style="border-left:3px solid #C26D66;">
                    <div style="font-size:0.82rem; font-weight:700; color:#9E4640;">65% — 100% : POTENTIALLY MANIPULATED</div>
                    <div style="font-size:0.78rem; color:var(--text-secondary);">Elevated statistical distortion or generative synthesis artifacts detected across primary feature families.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        s = backend.status()
        st.markdown(
            f"""
            <div class="fx-card">
                <div class="fx-card-header">
                    <span class="fx-card-title">Active Telemetry Nodes</span>
                </div>
                <div style="display:flex; flex-direction:column; gap:8px; font-size:0.85rem;">
                    <div style="display:flex; justify-content:space-between; padding-bottom:6px; border-bottom:1px solid var(--border-subtle);">
                        <span style="color:var(--text-secondary);">Engine Mode</span>
                        <b>{backend.mode.upper()}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding-bottom:6px; border-bottom:1px solid var(--border-subtle);">
                        <span style="color:var(--text-secondary);">Image Node</span>
                        <b style="color:var(--accent-sage);">{s['production'].get('image') or 'Unloaded'}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding-bottom:6px; border-bottom:1px solid var(--border-subtle);">
                        <span style="color:var(--text-secondary);">Audio Node</span>
                        <b style="color:var(--accent-sage);">{s['production'].get('audio') or 'Unloaded'}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:var(--text-secondary);">Video Node</span>
                        <b style="color:var(--accent-sage);">{s['production'].get('video') or 'Unloaded'}</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
