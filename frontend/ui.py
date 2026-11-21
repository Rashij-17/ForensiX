"""
ForensiX UI Design System & Component Primitives.
Deep Muted Slate Dark Theme Architecture with Layered Surfaces and 8px rhythm.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Calibrated Dark Mode Verdict Palette
VERDICT_COLORS = {
    "authentic": "#7A9A8B",     # Soft Sage Green
    "inconclusive": "#C89B6A",  # Warm Amber
    "manipulated": "#C26D66",   # Muted Terracotta Rosewood
}

VERDICT_BG = {
    "authentic": "rgba(122, 154, 139, 0.16)",
    "inconclusive": "rgba(200, 155, 106, 0.16)",
    "manipulated": "rgba(194, 109, 102, 0.16)",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700&display=swap');

:root {
    --bg-base: #0F1115;
    --surface-card: #161920;
    --surface-subtle: #1E222B;
    --surface-nav: rgba(22, 25, 32, 0.95);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(122, 154, 139, 0.45);
    --text-primary: #ECEEF2;
    --text-secondary: #959CA8;
    --text-tertiary: #6E7582;
    --accent-sage: #7A9A8B;
    --accent-sage-hover: #8DB3A2;
    --accent-sage-light: rgba(122, 154, 139, 0.18);
    --accent-teal: #6B9099;
    --accent-lavender: #8E8BA7;
    --verdict-manip: #C26D66;
    --verdict-inconc: #C89B6A;
    --radius-card: 16px;
    --radius-subcard: 12px;
    --radius-input: 10px;
    --radius-pill: 9999px;
    --ease-spring: cubic-bezier(0.22, 1, 0.36, 1);
    --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.35), 0 2px 6px -1px rgba(0, 0, 0, 0.2);
    --shadow-hover: 0 8px 28px -4px rgba(0, 0, 0, 0.5), 0 4px 12px -2px rgba(0, 0, 0, 0.3);
}

/* ================= REMOVE BLANK HEADER OVERLAY ================= */
header[data-testid="stHeader"],
header {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    overflow: hidden !important;
    visibility: hidden !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

/* ================= VIEWPORT & COMFORTABLE PADDING ================= */
.stApp {
    background-color: var(--bg-base) !important;
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.block-container {
    max-width: 1260px !important;
    padding-top: 1.4rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 1.75rem !important;
    padding-right: 1.75rem !important;
    margin-top: 0 !important;
}

/* ================= TYPOGRAPHY ================= */
h1, h2, h3, h4, h5 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.025em !important;
    margin-top: 0 !important;
}

h1 { font-weight: 800 !important; font-size: 2rem !important; line-height: 1.2 !important; }
h2 { font-weight: 750 !important; font-size: 1.55rem !important; line-height: 1.25 !important; }
h3 { font-weight: 700 !important; font-size: 1.25rem !important; }
h4 { font-weight: 650 !important; font-size: 1.05rem !important; color: var(--text-primary) !important; }

p, span, label, div {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

strong, b {
    color: var(--text-primary) !important;
    font-weight: 600;
}

code {
    background-color: var(--surface-subtle) !important;
    color: #D2D7E0 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    padding: 3px 7px !important;
    font-size: 0.85em !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

/* ================= COMPONENT: HERO BANNER & IMAGES ================= */
.fx-hero {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-card);
    padding: 28px 32px;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
    animation: fxFadeIn 0.35s var(--ease-spring);
}

div[data-testid="stImage"] img {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    transition: transform 0.28s var(--ease-spring) !important;
}

div[data-testid="stImage"] img:hover {
    transform: translateY(-2px) scale(1.01) !important;
}

.fx-hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent-sage);
    background: var(--accent-sage-light);
    border: 1px solid rgba(122, 154, 139, 0.3);
    border-radius: var(--radius-pill);
    padding: 3px 12px;
    margin-bottom: 10px;
}

.fx-hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin: 0 0 8px 0;
    line-height: 1.2;
}

.fx-hero-desc {
    font-size: 0.95rem;
    color: var(--text-secondary);
    line-height: 1.65;
    max-width: 720px;
    margin: 0;
}

/* ================= COMPONENT: CARDS & CONTAINERS ================= */
.fx-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-card);
    padding: 22px 26px;
    box-shadow: var(--shadow-card);
    margin-bottom: 1.25rem;
    transition: transform 0.28s var(--ease-spring), box-shadow 0.28s var(--ease-spring), border-color 0.28s var(--ease-spring);
    animation: fxFadeIn 0.35s var(--ease-spring);
}

.fx-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-hover);
    transform: translateY(-2px);
}

.fx-card-subtle {
    background: var(--surface-subtle);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-subcard);
    padding: 16px 20px;
    margin-bottom: 1rem;
    transition: border-color 0.2s var(--ease-spring);
}

.fx-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.fx-card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.015em;
    margin: 0;
}

.fx-card-chip {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent-sage);
    background: var(--accent-sage-light);
    border: 1px solid rgba(122, 154, 139, 0.25);
    border-radius: var(--radius-pill);
    padding: 2px 10px;
}

/* ================= COMPONENT: PILLS, BADGES, STATUS ================= */
.fx-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: var(--radius-pill);
    font-weight: 700;
    font-size: 0.76rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.fx-stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--surface-subtle);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill);
    padding: 5px 14px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    transition: all 0.2s var(--ease-spring);
}

.fx-stat-pill:hover {
    border-color: rgba(255, 255, 255, 0.16);
    color: var(--text-primary);
}

.fx-stat-pill b {
    color: var(--text-primary) !important;
}

/* ================= BUTTONS ================= */
.stButton > button,
.stDownloadButton > button,
button[data-testid="baseButton-secondary"] {
    background: #1C2029 !important;
    color: #ECEEF2 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: var(--radius-input) !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    transition: all 0.22s var(--ease-spring) !important;
    cursor: pointer !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
button[data-testid="baseButton-secondary"]:hover {
    background: #252B37 !important;
    border-color: rgba(122, 154, 139, 0.5) !important;
    color: #FFFFFF !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.4), 0 0 12px rgba(122, 154, 139, 0.18) !important;
}

.stButton > button:active,
.stDownloadButton > button:active {
    transform: translateY(0px) scale(0.98) !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4) !important;
}

.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #7A9A8B 0%, #688C7D 100%) !important;
    border: 1px solid rgba(122, 154, 139, 0.7) !important;
    color: #0F1115 !important;
    font-weight: 750 !important;
    font-size: 0.92rem !important;
    letter-spacing: -0.01em !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.3), 0 4px 16px rgba(122, 154, 139, 0.38) !important;
    transition: all 0.22s var(--ease-spring) !important;
}

.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #8DB3A2 0%, #7A9A8B 100%) !important;
    border-color: #8DB3A2 !important;
    color: #0A0D10 !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45), 0 8px 24px rgba(122, 154, 139, 0.55) !important;
    transform: translateY(-2px) !important;
}

.stButton > button[kind="primary"]:active,
button[data-testid="baseButton-primary"]:active {
    transform: translateY(0px) scale(0.98) !important;
    box-shadow: 0 2px 8px rgba(122, 154, 139, 0.3) !important;
}

.stDownloadButton > button {
    border-color: rgba(107, 144, 153, 0.35) !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent-teal) !important;
    box-shadow: 0 4px 16px rgba(107, 144, 153, 0.25) !important;
}

/* ================= INPUTS & FORMS ================= */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stDateInput input {
    background: var(--surface-subtle) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-input) !important;
    font-size: 0.92rem !important;
    padding: 10px 14px !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.25) !important;
    transition: border-color 0.2s var(--ease-spring), box-shadow 0.2s var(--ease-spring) !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: var(--accent-sage) !important;
    box-shadow: 0 0 0 3px rgba(122, 154, 139, 0.25) !important;
}

.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: var(--surface-subtle) !important;
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-input) !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s var(--ease-spring) !important;
}

.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: var(--border-hover) !important;
}

/* ================= METRIC TILES ================= */
[data-testid="stMetric"] {
    background: var(--surface-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-card) !important;
    padding: 18px 22px !important;
    box-shadow: var(--shadow-card) !important;
    transition: transform 0.28s var(--ease-spring), border-color 0.28s var(--ease-spring), box-shadow 0.28s var(--ease-spring) !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    border-color: var(--border-hover) !important;
    box-shadow: var(--shadow-hover) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 750 !important;
    font-size: 1.65rem !important;
    letter-spacing: -0.02em !important;
}

/* ================= TABS ================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: var(--surface-subtle) !important;
    padding: 6px !important;
    border-radius: var(--radius-card) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
    transition: all 0.2s var(--ease-spring) !important;
    border: 1px solid transparent !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(255, 255, 255, 0.04) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--surface-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35) !important;
    font-weight: 600 !important;
}

/* ================= FILE UPLOADER ================= */
[data-testid="stFileUploader"] {
    background: var(--surface-card) !important;
    border: 1.5px dashed rgba(122, 154, 139, 0.35) !important;
    border-radius: var(--radius-card) !important;
    padding: 22px !important;
    transition: border-color 0.25s var(--ease-spring), background 0.25s var(--ease-spring) !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-sage) !important;
    background: #181C25 !important;
}

/* ================= TABLES & ALERTS ================= */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-subcard) !important;
    background: var(--surface-card) !important;
    overflow: hidden !important;
}

[data-testid="stAlert"] {
    border-radius: var(--radius-subcard) !important;
    border: 1px solid var(--border-subtle) !important;
    background: var(--surface-card) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
}

.fx-warn {
    border-left: 3px solid var(--verdict-inconc);
    background: rgba(200, 155, 106, 0.08);
    border-top: 1px solid rgba(200, 155, 106, 0.22);
    border-right: 1px solid rgba(200, 155, 106, 0.22);
    border-bottom: 1px solid rgba(200, 155, 106, 0.22);
    padding: 14px 18px;
    border-radius: var(--radius-subcard);
    margin-bottom: 1.25rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--text-primary);
}

.fx-warn b {
    color: #E2B685 !important;
}

/* ================= SEGMENTED CONTROL / TOP NAV (WITH GENEROUS SPACING) ================= */
div[data-testid="stSegmentedControl"],
div[data-testid="stPills"] {
    background: transparent !important;
    width: 100% !important;
    padding: 0 4px !important;
}

div[data-testid="stSegmentedControl"] > div,
div[data-testid="stPills"] > div {
    background: rgba(22, 25, 32, 0.95) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: var(--radius-pill) !important;
    padding: 4px 6px !important;
    gap: 3px !important;
    box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.4) !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100% !important;
}

div[data-testid="stSegmentedControl"] button,
div[data-testid="stPills"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-pill) !important;
    color: #959CA8 !important;
    font-weight: 550 !important;
    font-size: 0.84rem !important;
    padding: 6px 14px !important;
    transition: all 0.22s var(--ease-spring) !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    flex: 1 1 auto !important;
    text-align: center !important;
}

div[data-testid="stSegmentedControl"] button:hover,
div[data-testid="stPills"] button:hover {
    background: #1E222B !important;
    color: #ECEEF2 !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[data-checked="true"],
div[data-testid="stPills"] button[aria-checked="true"],
div[data-testid="stPills"] button[data-checked="true"] {
    background: #1E222B !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(122, 154, 139, 0.5) !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4), 0 0 12px rgba(122, 154, 139, 0.22) !important;
    font-weight: 650 !important;
}

/* ================= RADIO BUTTONS FALLBACK ================= */
div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: var(--surface-subtle) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-pill) !important;
    padding: 6px 16px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.22s var(--ease-spring) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: #252B37 !important;
    border-color: rgba(122, 154, 139, 0.4) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(122, 154, 139, 0.18) !important;
    border: 1px solid var(--accent-sage) !important;
    box-shadow: 0 0 14px rgba(122, 154, 139, 0.25) !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ================= KEYFRAME ANIMATIONS ================= */
@keyframes fxFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulseDot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(122, 154, 139, 0.6); }
    50% { box-shadow: 0 0 0 6px rgba(122, 154, 139, 0); }
}

footer { visibility: hidden; }
</style>
"""


def setup_page():
    st.set_page_config(
        page_title="ForensiX — Media Authenticity Intelligence",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tag: str = "Forensic Workspace"):
    st.markdown(
        f"""
        <div class="fx-hero">
            <div class="fx-hero-tag">{tag}</div>
            <div class="fx-hero-title">{title}</div>
            <p class="fx-hero-desc">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str, text: str | None = None) -> str:
    color = VERDICT_COLORS.get(label, "#959CA8")
    bg = VERDICT_BG.get(label, "rgba(149, 156, 168, 0.15)")
    display = (text or label).upper()
    return f'<span class="fx-badge" style="color:{color}; background:{bg}; border: 1px solid {color}40;">{display}</span>'


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def gauge(p: float, title: str = "Manipulation Probability") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=100 * p,
            number={
                "suffix": "%",
                "valueformat": ".1f",
                "font": {"color": "#ECEEF2", "family": "Plus Jakarta Sans, sans-serif", "size": 36, "weight": 700},
            },
            title={"text": title, "font": {"size": 13, "color": "#959CA8", "family": "Plus Jakarta Sans, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6E7582", "tickfont": {"color": "#959CA8", "size": 10}},
                "bar": {"color": "#7A9A8B", "thickness": 0.24},
                "bgcolor": "#1E222B",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.08)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(122, 154, 139, 0.25)"},
                    {"range": [35, 65], "color": "rgba(200, 155, 106, 0.22)"},
                    {"range": [65, 100], "color": "rgba(194, 109, 102, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#959CA8", "family": "Plus Jakarta Sans, -apple-system, sans-serif"},
    )
    return fig


def shap_chart(contribs: list[dict]) -> go.Figure:
    items = list(reversed(contribs))
    fig = go.Figure(
        go.Bar(
            x=[i["shap"] for i in items],
            y=[i["feature"] for i in items],
            orientation="h",
            marker_color=["#C26D66" if i["shap"] > 0 else "#7A9A8B" for i in items],
            marker_line=dict(width=0),
            customdata=[[i["description"], i["value"]] for i in items],
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>value: %{customdata[1]:.4g}<br>SHAP: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=60 + 32 * len(items),
        margin=dict(l=10, r=10, t=30, b=40),
        xaxis_title="SHAP Value (Right: pushes manipulated · Left: pushes authentic)",
        title=dict(text="Feature Attribution Breakdown", font=dict(color="#ECEEF2", size=13, weight=600)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#959CA8", family="Plus Jakarta Sans, -apple-system, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            tickfont=dict(color="#959CA8", size=10),
            title_font=dict(color="#959CA8", size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            tickfont=dict(color="#ECEEF2", size=11),
        ),
    )
    fig.add_vline(x=0, line_width=1, line_color="rgba(122, 154, 139, 0.5)")
    return fig


def b64_image(b64: str, caption: str | None = None, **kw):
    st.image(base64.b64decode(b64), caption=caption, **kw)


def media_preview(data: bytes, name: str, modality: str):
    if modality == "image":
        st.image(data, caption=name, width="stretch")
    elif modality == "audio":
        st.audio(data)
    else:
        st.video(data)


# --------------------------------------------------
# Shared File State
# --------------------------------------------------
def set_current(name: str, data: bytes, result: dict | None = None):
    st.session_state["current"] = {
        "name": name,
        "data": data,
        "result": result,
        "hash": hashlib.sha256(data).hexdigest(),
    }


def current() -> dict | None:
    return st.session_state.get("current")


def pick_file(label: str, key: str, types=None, allow_current: bool = True):
    cur = current() if allow_current else None
    use_cur = False
    if cur:
        use_cur = st.toggle(f"Use file from Analyze: {cur['name']}", value=True, key=f"{key}_usecur")
    if use_cur:
        return cur["name"], cur["data"]
    up = st.file_uploader(label, type=types, key=f"{key}_up")
    return (up.name, up.getvalue()) if up else (None, None)


EXT = [
    "jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff",
    "wav", "mp3", "flac", "ogg", "m4a",
    "mp4", "avi", "mov", "mkv", "webm",
]


def df_show(rows, height=None, **kw):
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Nothing to show yet.")
        return df
    st.dataframe(df, width="stretch", hide_index=True, height=height, **kw)
    return df


def need_models(backend) -> bool:
    s = backend.status()
    if not s["ready"]:
        st.error(
            "No trained models were found. Run the training pipeline first: `make demo` (synthetic demo data) "
            "or `make pipeline` (after placing the DeepSafe dataset in data/raw)."
        )
        return False
    return True
