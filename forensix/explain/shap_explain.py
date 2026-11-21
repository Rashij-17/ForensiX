"""SHAP explainability for the feature-based models (global importance + per-sample explanations)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.catalog import describe_feature, indicator_group


def _positive_class(sv) -> np.ndarray:
    """Normalise the many SHAP output shapes to (n_samples, n_features) for the 'manipulated' class."""
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    return sv


def build_explainer(model, algorithm: str, background: pd.DataFrame):
    """TreeExplainer for tree models, a probability-space Permutation/Kernel explainer for the rest."""
    import shap

    if algorithm in ("random_forest", "xgboost"):
        return shap.TreeExplainer(model), "tree"
    fn = lambda X: model.predict_proba(pd.DataFrame(X, columns=background.columns))[:, 1]  # noqa: E731
    return shap.Explainer(fn, background, algorithm="permutation", seed=0), "permutation"


def _explain_raw(explainer, kind: str, X: pd.DataFrame):
    if kind == "tree":
        sv = _positive_class(explainer.shap_values(X.to_numpy(float)))
        base = explainer.expected_value
        base = float(np.ravel(base)[-1])
        return sv, base
    ex = explainer(X.to_numpy(float), max_evals=2 * X.shape[1] + 1, silent=True)
    return np.asarray(ex.values), float(np.mean(ex.base_values))


def explain_sample(
    model, algorithm: str, background: pd.DataFrame, x: pd.Series, top_k: int = 8, reference: dict | None = None
) -> dict:
    """Ranked contribution list for ONE sample: direction, size, plain-language meaning, context vs reference."""
    explainer, kind = build_explainer(model, algorithm, background)
    X = pd.DataFrame([x.to_numpy(float)], columns=x.index)
    sv, base = _explain_raw(explainer, kind, X)
    contrib = sv[0]
    order = np.argsort(-np.abs(contrib))
    items = []
    for i in order[:top_k]:
        name = x.index[i]
        grp, desc = describe_feature(name)
        val = float(x.iloc[i])
        item = {
            "feature": name,
            "group": grp,
            "description": desc,
            "value": val,
            "shap": float(contrib[i]),
            "direction": "manipulated" if contrib[i] > 0 else "authentic",
        }
        if reference and name in reference:
            r, f = reference[name]["real"], reference[name]["fake"]
            item["real_mean"], item["fake_mean"] = r["mean"], f["mean"]
            item["closer_to"] = (
                "manipulated" if abs(val - f["mean"]) / f["std"] < abs(val - r["mean"]) / r["std"] else "authentic"
            )
        items.append(item)
    by_group: dict[str, float] = {}
    for n, c in zip(x.index, contrib):
        g = indicator_group(n).replace(" (per-frame)", "")
        by_group[g] = by_group.get(g, 0.0) + float(c)
    return {
        "base_value": base,
        "output_value": base + float(contrib.sum()),
        "contributions": items,
        "group_totals": dict(sorted(by_group.items(), key=lambda kv: -abs(kv[1]))),
        "all_shap": {n: float(c) for n, c in zip(x.index, contrib)},
        "method": kind,
    }


def explain_global(
    model, algorithm: str, background: pd.DataFrame, X: pd.DataFrame, max_samples: int = 150, seed: int = 0
) -> dict:
    """Mean |SHAP| ranking and the raw SHAP matrix (for beeswarm / bar plots)."""
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=seed)
    explainer, kind = build_explainer(model, algorithm, background)
    sv, base = _explain_raw(explainer, kind, X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return {
        "importance": imp,
        "shap_values": sv,
        "X": X,
        "base_value": base,
        "groups": imp.groupby([indicator_group(n).replace(" (per-frame)", "") for n in imp.index])
        .sum()
        .sort_values(ascending=False),
    }
