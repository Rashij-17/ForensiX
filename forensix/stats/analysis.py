"""Statistical validation: do the engineered features really differ between real and manipulated media?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from ..data.dataset import feature_columns, load_feature_table
from ..features.catalog import describe_feature
from ..utils import get_logger, write_json

log = get_logger()


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else 0.0


def _ci_mean_diff(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Welch confidence interval for mean(fake) - mean(real)."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return 0.0, 0.0
    dfree = (va + vb) ** 2 / ((va**2) / (len(a) - 1) + (vb**2) / (len(b) - 1))
    t = sps.t.ppf(0.5 + level / 2, dfree)
    d = b.mean() - a.mean()
    return float(d - t * se), float(d + t * se)


def _outlier_share(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return float(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0


def feature_tests(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-feature descriptive statistics + hypothesis tests (real = 0 vs fake = 1)."""
    real, fake = df[df["label"] == 0], df[df["label"] == 1]
    rows = []
    for f in features:
        a, b = real[f].to_numpy(float), fake[f].to_numpy(float)
        if len(a) < 3 or len(b) < 3 or (a.std() == 0 and b.std() == 0):
            continue
        t_stat, t_p = sps.ttest_ind(a, b, equal_var=False)
        u_stat, u_p = sps.mannwhitneyu(a, b, alternative="two-sided")
        ks_stat, ks_p = sps.ks_2samp(a, b)
        lo, hi = _ci_mean_diff(a, b)
        grp, desc = describe_feature(f)
        rows.append(
            {
                "feature": f,
                "group": grp,
                "description": desc,
                "n_real": len(a),
                "n_fake": len(b),
                "mean_real": a.mean(),
                "mean_fake": b.mean(),
                "std_real": a.std(ddof=1),
                "std_fake": b.std(ddof=1),
                "median_real": np.median(a),
                "median_fake": np.median(b),
                "skew_real": float(sps.skew(a)),
                "skew_fake": float(sps.skew(b)),
                "diff_mean": b.mean() - a.mean(),
                "ci95_low": lo,
                "ci95_high": hi,
                "cohens_d": _cohens_d(a, b),
                "auc_univariate": float(u_stat / (len(a) * len(b))),
                "welch_t": float(t_stat),
                "welch_p": float(t_p),
                "mannwhitney_p": float(u_p),
                "ks_stat": float(ks_stat),
                "ks_p": float(ks_p),
                "outlier_share_real": _outlier_share(a),
                "outlier_share_fake": _outlier_share(b),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Mann-Whitney U measures P(real > fake); express direction as P(fake > real)
    out["auc_univariate"] = 1.0 - out["auc_univariate"]
    for col in ("welch_p", "mannwhitney_p", "ks_p"):
        out[col.replace("_p", "_q")] = multipletests(out[col].fillna(1.0), method="fdr_bh")[1]
    out["significant"] = out["mannwhitney_q"] < 0.05
    out["abs_d"] = out["cohens_d"].abs()
    return out.sort_values("abs_d", ascending=False).reset_index(drop=True)


def correlation_study(df: pd.DataFrame, features: list[str], threshold: float = 0.9) -> dict:
    """Spearman correlation matrix, covariance summary and highly redundant feature pairs."""
    X = df[features]
    corr = X.corr(method="spearman").fillna(0.0)
    cov = X.cov()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iat[i, j]) >= threshold:
                pairs.append({"a": cols[i], "b": cols[j], "spearman": float(corr.iat[i, j])})
    pairs.sort(key=lambda p: -abs(p["spearman"]))
    return {"corr": corr, "cov_diag": cov.values.diagonal().tolist(), "redundant_pairs": pairs}


def run_statistics(cfg, modality: str) -> dict:
    """Stage 5: run the full statistical validation on the TRAIN split and save the results."""
    df = load_feature_table(cfg, modality)
    train = df[df["split"] == "train"]
    feats = feature_columns(df)
    tests = feature_tests(train, feats)
    cor = correlation_study(train, feats)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(cfg.reports_dir / f"stats_{modality}.csv", index=False)
    cor["corr"].to_csv(cfg.reports_dir / f"correlation_{modality}.csv")
    n_sig = int(tests["significant"].sum()) if not tests.empty else 0
    summary = {
        "modality": modality,
        "n_train": int(len(train)),
        "n_features_tested": int(len(tests)),
        "n_significant_fdr05": n_sig,
        "share_significant": float(n_sig / max(len(tests), 1)),
        "strong_effect_features": tests[tests["abs_d"] >= 0.8]["feature"].tolist() if not tests.empty else [],
        "top_features": (
            tests.head(10)[["feature", "cohens_d", "mannwhitney_q"]].to_dict("records") if not tests.empty else []
        ),
        "n_redundant_pairs": len(cor["redundant_pairs"]),
        "redundant_pairs": cor["redundant_pairs"][:15],
        "method": "Welch t-test, Mann-Whitney U, two-sample KS; Benjamini-Hochberg FDR across features; "
        "Cohen's d effect size; 95% Welch CI of mean difference; IQR outlier share; Spearman correlation.",
    }
    write_json(cfg.reports_dir / f"stats_{modality}.json", summary)
    log.info(
        "[%s] %d/%d features differ significantly (FDR 5%%); %d with |d|>=0.8",
        modality,
        n_sig,
        len(tests),
        len(summary["strong_effect_features"]),
    )
    return summary
