"""Forensic report generation (Markdown + self-contained HTML) in calibrated, non-absolute language."""

from __future__ import annotations

import datetime as dt
import html
import pandas as pd

from . import __version__
from .inference import LIMITATIONS
from .utils import fig_to_b64, read_json

ASSESSMENT_TEXT = {
    "manipulated": "The analysed {modality} is **potentially manipulated or AI-generated**. The forensic indicators "
    "lean towards synthetic or altered content, but this is a statistical estimate, not proof.",
    "authentic": "The analysed {modality} shows **no strong indication of manipulation**. This does not prove that it "
    "is authentic; it means the measured indicators resemble those of the authentic samples the model saw.",
    "inconclusive": "The analysis is **inconclusive**. The indicators are mixed and the model does not lean clearly "
    "either way. Additional evidence (provenance, metadata, a different version of the file, "
    "human review) is needed before drawing any conclusion.",
}


def _pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def robustness_context(cfg, modality: str) -> str | None:
    p = cfg.reports_dir / f"robustness_{modality}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    deg = d[d["transformation"] != "Clean"].dropna(subset=["f1_drop"])
    if deg.empty:
        return None
    clean = float(d.loc[d["transformation"] == "Clean", "f1"].iloc[0])
    w = deg.sort_values("f1_drop", ascending=False).iloc[0]
    return (
        f"On the held-out test set the {modality} model reached F1 {clean:.2f} on unmodified media. "
        f"The largest measured degradation was under '{w['transformation']}' (F1 {w['f1']:.2f}, a drop of "
        f"{w['f1_drop']:.2f}). Results for media that has been compressed, resized or re-shared are therefore "
        f"less reliable than this figure suggests."
    )


def generator_context(cfg, modality: str) -> str | None:
    j = read_json(cfg.reports_dir / f"generators_{modality}.json")
    if not j or j.get("spread") is None:
        return None
    return (
        f"Across the generators in the benchmark the detection rate varied by {j['spread']:.2f} "
        f"(weakest: {j['weakest_source']}, strongest: {j['strongest_source']}). The detector therefore does not "
        f"perform equally well on every kind of synthetic media."
    )


def build_report(result: dict, cfg, case: dict | None = None, analyst: str = "", figures: dict | None = None) -> dict:
    """Return {markdown, html, json}. ``figures`` maps caption -> matplotlib figure (embedded in the HTML only)."""
    v, m = result["verdict"], result["model"]
    p = result["probability_manipulated"]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    expl = result.get("explanation", {})
    contrib = expl.get("contributions", [])
    md: list[str] = []
    md.append("# ForensiX Forensic Report")
    md.append(f"*Generated {now} by ForensiX v{__version__}. Decision-support output; see Limitations.*\n")
    md.append("## 1. Case information")
    md.append(f"- **Case:** {case['id'] + ' - ' + case['name'] if case else 'Stand-alone analysis'}")
    if analyst:
        md.append(f"- **Analyst:** {analyst}")
    md.append(f"- **File:** {result.get('filename') or 'n/a'}")
    md.append(f"- **SHA-256:** `{result.get('sha256') or 'n/a'}`")
    md.append(f"- **Modality:** {result['modality']}")
    md.append(f"- **Model:** {m['display']} (dataset {m.get('dataset_version') or 'n/a'})\n")
    md.append("## 2. Authenticity assessment")
    md.append(
        f"**Outcome: {v['text']}.**  Probability of manipulation: **{_pct(p)}** "
        f"(authentic {_pct(1 - p)}). Strength of evidence: {v['evidence_strength']}.\n"
    )
    md.append(ASSESSMENT_TEXT[v["label"]].format(modality=result["modality"]) + "\n")
    tm = m.get("test_metrics") or {}
    if tm:
        md.append(
            f"*Model quality on the held-out test set: accuracy {tm.get('test_accuracy', float('nan')):.2f}, "
            f"F1 {tm.get('test_f1', float('nan')):.2f}, ROC AUC {tm.get('test_roc_auc', float('nan')):.2f}. "
            f"These describe the benchmark data, not this file.*\n"
        )
    md.append("## 3. Model results")
    md.append("| Model | P(manipulated) | Role |\n|---|---|---|")
    for r in result.get("model_results", []):
        md.append(
            f"| {r['model']} | {_pct(r['probability_manipulated'])} | {'production' if r['production'] else r['stage']} |"
        )
    if result.get("model_results"):
        ps = [r["probability_manipulated"] for r in result["model_results"]]
        agree = all((x >= 0.5) == (ps[0] >= 0.5) for x in ps)
        md.append(f"\nThe candidate models {'agree' if agree else 'DISAGREE'} on the direction of the result.\n")
    md.append("## 4. Key forensic indicators and explanation (SHAP)")
    if contrib:
        md.append(
            "Ranked by influence on this prediction. Positive SHAP values push towards *manipulated*, negative towards *authentic*.\n"
        )
        md.append("| # | Indicator | Group | Value | Effect | SHAP |\n|---|---|---|---|---|---|")
        for i, c in enumerate(contrib, 1):
            md.append(
                f"| {i} | `{c['feature']}` - {c['description']} | {c['group']} | {c['value']:.4g} | "
                f"towards {c['direction']} | {c['shap']:+.3f} |"
            )
        gt = expl.get("group_totals", {})
        if gt:
            top = list(gt.items())[:3]
            md.append("\nBy indicator group: " + "; ".join(f"{g} ({s:+.2f})" for g, s in top) + ".")
    else:
        md.append("Explanations were not computed for this analysis.")
    md.append("\n## 5. Context and reliability")
    for ctx in (robustness_context(cfg, result["modality"]), generator_context(cfg, result["modality"])):
        if ctx:
            md.append(f"- {ctx}")
    md.append("\n## 6. Limitations of this assessment")
    for lim in LIMITATIONS:
        md.append(f"- {lim}")
    md.append("\n## 7. Data handling")
    md.append(
        "- The uploaded media was processed only for this analysis and deleted afterwards; only its SHA-256, "
        "extracted numeric features and this report are retained."
    )
    markdown = "\n".join(md)
    html_doc = _to_html(markdown, figures or {}, v["label"])
    return {
        "markdown": markdown,
        "html": html_doc,
        "json": {
            k: result[k] for k in ("filename", "sha256", "modality", "probability_manipulated", "verdict", "model")
        },
    }


def case_report(case: dict, cfg) -> dict:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    s = case["stats"]
    md = [
        f"# ForensiX Case Report: {case['id']} - {case['name']}",
        f"*Generated {now}. Status: {case['status']}.*\n",
        f"{case.get('description') or ''}\n",
        "## Summary",
        f"- Analyses: {s['n_analyses']}; potentially manipulated: {s['n_flagged']}; inconclusive: {s['n_inconclusive']}",
    ]
    if s["mean_probability"] is not None:
        md.append(
            f"- Mean probability of manipulation: {_pct(s['mean_probability'])}; highest: {_pct(s['max_probability'])}"
        )
    md += [
        "\n## Analyses",
        "| File | Modality | P(manipulated) | Outcome | Model | Top indicators |",
        "|---|---|---|---|---|---|",
    ]
    for a in case["analyses"]:
        top = ", ".join(i["feature"] for i in a["summary"].get("top_indicators", [])[:3])
        md.append(
            f"| {a['filename']} | {a['modality']} | {_pct(a['probability'])} | {a['summary'].get('verdict_text', a['verdict'])} | {a['model']} | {top} |"
        )
    md += ["\n## Timeline"] + [f"- {e['ts']}: {e['event']}" for e in case["events"]]
    md += ["\n## Limitations"] + [f"- {x}" for x in LIMITATIONS]
    markdown = "\n".join(md)
    return {"markdown": markdown, "html": _to_html(markdown, {}, None)}


_CSS = """body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1d2433;line-height:1.55}
h1{border-bottom:3px solid #3a6ea5;padding-bottom:.3rem}h2{margin-top:1.8rem;color:#274c77}
table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{border:1px solid #d5dbe6;padding:.35rem .5rem;text-align:left}
th{background:#eef2f8}code{background:#eef2f8;padding:0 .25rem;border-radius:3px}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:1rem;color:#fff;font-weight:600}
.manipulated{background:#c8452b}.authentic{background:#2a9d8f}.inconclusive{background:#c89b2b}
figure{margin:1rem 0}figcaption{font-size:.85rem;color:#566}img{max-width:100%}"""


def _inline(t: str) -> str:
    t = html.escape(t)
    import re

    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def _to_html(markdown: str, figures: dict, label: str | None) -> str:
    out, lines, i = [], markdown.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            out.append(f"<h1>{_inline(ln[2:])}</h1>")
            if label:
                out.append(f'<p><span class="badge {label}">{html.escape(label.upper())}</span></p>')
        elif ln.startswith("## "):
            out.append(f"<h2>{_inline(ln[3:])}</h2>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            i -= 1
            cells = [
                [c.strip() for c in r.strip("|").split("|")]
                for r in rows
                if not set(r.replace("|", "").strip()) <= {"-", " ", ":"}
            ]
            t = "<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells[0]) + "</tr>"
            for r in cells[1:]:
                t += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
            out.append(t + "</table>")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            i -= 1
            out.append("<ul>" + "".join(items) + "</ul>")
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    for caption, fig in figures.items():
        out.append(
            f'<figure><img src="data:image/png;base64,{fig_to_b64(fig)}" alt="{html.escape(caption)}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )
    return f"<!doctype html><html><head><meta charset='utf-8'><title>ForensiX report</title><style>{_CSS}</style></head><body>{''.join(out)}</body></html>"
