"""
ForensiX - Investigations & Case Dossier Management
Chain of custody, dossier asset logs, and case reporting.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import ui


def render(backend):
    ui.hero(
        "Forensic Dossiers & Case Management",
        "Orchestrate multi-asset forensic investigations: aggregate evidence, track chronological chain-of-custody, "
        "and export comprehensive forensic dossiers.",
        tag="Case Intelligence",
    )

    with st.expander("Initialize New Investigation Dossier", expanded=not backend.list_cases()):
        with st.form("new_case", clear_on_submit=True):
            c_f1, c_f2 = st.columns([1.5, 1], gap="medium")
            with c_f1:
                name = st.text_input("Investigation Title")
                desc = st.text_area("Operational Scope & Objectives")
            with c_f2:
                analyst = st.text_input("Lead Examiner / Analyst Handle", placeholder="e.g. Lead Examiner #04")
            if st.form_submit_button("Create Investigation Dossier", type="primary"):
                try:
                    c = backend.create_case(name, desc, analyst)
                    st.session_state["case_sel"] = c["id"]
                    st.success(f"Dossier {c['id']} created successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    cases = backend.list_cases()
    if not cases:
        st.info("No active investigation dossiers on record.")
        return

    # KPI Tiles
    tot_cases = len(cases)
    tot_analyses = sum(c["n_analyses"] for c in cases)
    tot_flagged = sum(c["n_flagged"] for c in cases)
    k1, k2, k3 = st.columns(3, gap="medium")
    k1.metric("Active Case Dossiers", tot_cases)
    k2.metric("Total Items Evaluated", tot_analyses)
    k3.metric("Flagged Manipulations", tot_flagged)

    st.markdown('<div style="margin: 18px 0 12px 0;"></div>', unsafe_allow_html=True)
    st.markdown("#### Dossier Registry")

    df = pd.DataFrame(cases)[["id", "name", "status", "n_analyses", "n_flagged", "analyst", "updated"]].rename(
        columns={
            "id": "Case ID",
            "name": "Investigation Title",
            "status": "Status",
            "n_analyses": "Media Items",
            "n_flagged": "Flagged",
            "analyst": "Examiner",
            "updated": "Last Activity",
        }
    )
    st.dataframe(df, width="stretch", hide_index=True)

    ids = [c["id"] for c in cases]
    sel = st.selectbox(
        "Open Specific Dossier",
        ids,
        index=(
            ids.index(st.session_state.get("case_sel", ids[0]))
            if st.session_state.get("case_sel", ids[0]) in ids
            else 0
        ),
        format_func=lambda i: f"{i} — {next(c['name'] for c in cases if c['id'] == i)}",
    )
    st.session_state["case_sel"] = sel
    case = backend.get_case(sel)
    s = case["stats"]

    # Active Dossier Header
    st.markdown(
        f"""
        <div class="fx-card" style="border-left: 3px solid var(--accent-sage);">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <div style="font-size:1.25rem; font-weight:700; color:var(--text-primary);">{case['name']}</div>
                    <div style="font-size:0.8rem; color:var(--text-secondary); margin-top:3px;">
                        ID: <b>{case['id']}</b> · Lead Examiner: <b>{case['analyst']}</b> · Status: <span style="color:var(--accent-sage); text-transform:uppercase; font-weight:700;">{case['status']}</span>
                    </div>
                </div>
                <div style="display:flex; gap:8px;">
                    <span class="fx-stat-pill">Items: <b>{s['n_analyses']}</b></span>
                    <span class="fx-stat-pill">Flagged: <b style="color:var(--verdict-manip);">{s['n_flagged']}</b></span>
                    <span class="fx-stat-pill">Mean P: <b>{ui.pct(s['mean_probability'])}</b></span>
                </div>
            </div>
            {f'<div style="margin-top:12px; font-size:0.88rem; color:var(--text-secondary); border-top:1px solid var(--border-subtle); padding-top:10px;">{case["description"]}</div>' if case.get("description") else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.tabs(["Dossier Assets", "Append Evidence Asset", "Chronological Audit Trail & Governance"])
    with t1:
        if case["analyses"]:
            rows = [
                {
                    "Asset": x["filename"],
                    "Modality": x["modality"].upper(),
                    "P(manipulated)": round(x["probability"], 3),
                    "Finding": x["summary"].get("verdict_text", x["verdict"]),
                    "Evaluator": x["model"],
                    "Primary Indicators": ", ".join(i["feature"] for i in x["summary"].get("top_indicators", [])[:3]),
                    "Timestamp": x["created"],
                }
                for x in case["analyses"]
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No media items archived in this dossier yet.")
    with t2:
        if case["status"] == "closed":
            st.warning("This dossier is closed. Re-open status under Governance to append further evaluations.")
        else:
            up = st.file_uploader("Select asset to append to dossier", type=ui.EXT, key=f"case_up_{sel}")
            note = st.text_input("Analytical note / Chain of custody remarks", key=f"case_note_{sel}")
            if up and st.button("Analyze & Append to Dossier", type="primary"):
                with st.spinner("Executing forensic pipeline and archiving findings..."):
                    try:
                        res = backend.analyze(up.getvalue(), up.name, sel, True, note)
                        ui.set_current(up.name, up.getvalue(), res)
                        st.success(f"Appended {up.name}: {res['verdict']['text']} (P={ui.pct(res['probability_manipulated'])})")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
    with t3:
        left, right = st.columns([1.6, 1], gap="medium")
        with left:
            st.markdown("#### Chronological Audit Trail")
            for e in reversed(case["events"]):
                st.markdown(
                    f"<div class='fx-card-subtle' style='margin-bottom:8px; padding:10px 14px;'>"
                    f"<span style='color:var(--text-tertiary); font-family:monospace; font-size:0.78rem;'>{e['ts']}</span> &nbsp; "
                    f"<span style='color:var(--text-primary); font-size:0.86rem;'>{e['event']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown("#### Governance Actions")
            status = st.selectbox(
                "Dossier Status",
                ["open", "under review", "closed"],
                index=["open", "under review", "closed"].index(case["status"]),
            )
            if status != case["status"] and st.button("Update Status"):
                backend.set_case_status(sel, status)
                st.rerun()
            txt = st.text_input("Append Examiner Note", key=f"note2_{sel}")
            if txt and st.button("Archive Note"):
                backend.add_case_note(sel, txt)
                st.rerun()

            st.markdown('<div style="margin: 16px 0; border-top: 1px solid var(--border-subtle);"></div>', unsafe_allow_html=True)
            st.markdown("#### Export Case Dossier")
            rep = backend.case_report(sel)
            st.download_button("Export Dossier (HTML)", rep["html"], f"{sel}_forensic_dossier.html", "text/html", width="stretch")
            st.download_button("Export Dossier (Markdown)", rep["markdown"], f"{sel}_forensic_dossier.md", "text/markdown", width="stretch")
            if st.button("Purge Dossier", type="secondary"):
                backend.delete_case(sel)
                st.session_state.pop("case_sel", None)
                st.rerun()
