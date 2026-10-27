"""Inferential statistics: which features are associated with the target, and how strongly."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

from .profiling import BOOLEAN, CATEGORICAL, NUMERIC
from .target import CLASSIFICATION
from .utils import safe_float

MAX_GROUPS = 25
NORMALITY_SAMPLE = 5000

# Conventional effect-size thresholds (Cohen 1988 and common practice for eta squared).
_EFFECT_THRESHOLDS = {
    "r": (0.1, 0.3, 0.5),
    "cramers_v": (0.1, 0.3, 0.5),
    "eta_squared": (0.01, 0.06, 0.14),
}


def effect_label(value: float | None, measure: str) -> str:
    if value is None:
        return "n/a"
    small, medium, large = _EFFECT_THRESHOLDS[measure]
    value = abs(value)
    if value >= large:
        return "large"
    if value >= medium:
        return "medium"
    if value >= small:
        return "small"
    return "negligible"


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    """False-discovery-rate adjusted p-values (Benjamini-Hochberg step-up)."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and np.isfinite(p)]
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    m = len(indexed)
    indexed.sort(key=lambda t: t[1])
    running_min = 1.0
    for rank in range(m, 0, -1):
        i, p = indexed[rank - 1]
        running_min = min(running_min, p * m / rank)
        adjusted[i] = min(running_min, 1.0)
    return adjusted


def _top_groups(feature: pd.Series, limit: int = MAX_GROUPS) -> pd.Series:
    keep = feature.value_counts().head(limit).index
    return feature.where(feature.isin(keep))


def _anova(values: pd.Series, groups: pd.Series) -> dict | None:
    frame = pd.DataFrame({"v": values.astype(float), "g": groups.astype(str)}).dropna()
    samples = [g["v"].to_numpy() for _, g in frame.groupby("g") if len(g) >= 2]
    if len(samples) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_stat, p = sps.f_oneway(*samples)
    grand = frame["v"].mean()
    ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_total = ((frame["v"] - grand) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total else 0.0
    return {
        "test": "One-way ANOVA",
        "statistic": safe_float(f_stat),
        "p_value": safe_float(p),
        "effect_size": safe_float(eta2),
        "effect_measure": "eta_squared",
        "n": int(len(frame)),
    }


def _chi_square(a: pd.Series, b: pd.Series) -> dict | None:
    frame = pd.DataFrame({"a": a.astype(str), "b": b.astype(str)})[a.notna() & b.notna()]
    table = pd.crosstab(frame["a"], frame["b"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, p, dof, _ = sps.chi2_contingency(table)
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n and k else 0.0
    return {
        "test": "Chi-square independence",
        "statistic": safe_float(chi2),
        "p_value": safe_float(p),
        "dof": int(dof),
        "effect_size": safe_float(cramers_v),
        "effect_measure": "cramers_v",
        "n": int(n),
    }


def _correlation(a: pd.Series, b: pd.Series) -> dict | None:
    frame = pd.DataFrame({"a": a.astype(float), "b": b.astype(float)}).dropna()
    if len(frame) < 3 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return None
    r, p = sps.pearsonr(frame["a"], frame["b"])
    rho, p_s = sps.spearmanr(frame["a"], frame["b"])
    return {
        "test": "Pearson correlation",
        "statistic": safe_float(r),
        "p_value": safe_float(p),
        "spearman_rho": safe_float(rho),
        "spearman_p": safe_float(p_s),
        "effect_size": safe_float(r),
        "effect_measure": "r",
        "n": int(len(frame)),
    }


def association_tests(df: pd.DataFrame, target: str, task: str, semantic: dict[str, str], alpha: float = 0.05) -> dict:
    """Test every usable feature against the target and correct for multiple comparisons."""
    target_is_categorical = task == CLASSIFICATION
    y = df[target]
    results = []
    for col in df.columns:
        if col == target:
            continue
        kind = semantic.get(col)
        if kind not in (NUMERIC, CATEGORICAL, BOOLEAN):
            continue
        x = df[col]
        feature_is_numeric = kind == NUMERIC
        if feature_is_numeric and target_is_categorical:
            res = _anova(x, _top_groups(y))
        elif feature_is_numeric:
            res = _correlation(x, y)
        elif target_is_categorical:
            res = _chi_square(_top_groups(x), _top_groups(y))
        else:
            res = _anova(y, _top_groups(x))
        if res is None:
            continue
        res["feature"] = col
        res["feature_type"] = kind
        res["effect"] = effect_label(res["effect_size"], res["effect_measure"])
        results.append(res)

    adjusted = benjamini_hochberg([r["p_value"] for r in results])
    for r, p_adj in zip(results, adjusted, strict=True):
        r["p_adjusted"] = p_adj
        r["significant"] = p_adj is not None and p_adj < alpha

    # Effect sizes use different scales (r, V, eta²), so rank by magnitude band before p-value.
    band = {"large": 3, "medium": 2, "small": 1, "negligible": 0, "n/a": -1}
    results.sort(
        key=lambda r: (not r["significant"], -band[r["effect"]], r["p_adjusted"] if r["p_adjusted"] is not None else 1)
    )
    return {
        "target": target,
        "task": task,
        "alpha": alpha,
        "correction": "Benjamini-Hochberg FDR",
        "tests": results,
        "significant_count": sum(r["significant"] for r in results),
    }


def normality_tests(df: pd.DataFrame, columns: list[str], alpha: float = 0.05) -> list[dict]:
    """D'Agostino-Pearson K² for n >= 20 (Shapiro-Wilk for smaller samples)."""
    out = []
    for col in columns:
        values = df[col].dropna().astype(float)
        if len(values) < 8 or values.nunique() < 3:
            continue
        if len(values) > NORMALITY_SAMPLE:
            values = values.sample(NORMALITY_SAMPLE, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(values) >= 20:
                stat, p = sps.normaltest(values)
                test = "D'Agostino-Pearson"
            else:
                stat, p = sps.shapiro(values)
                test = "Shapiro-Wilk"
        out.append(
            {
                "column": col,
                "test": test,
                "statistic": safe_float(stat),
                "p_value": safe_float(p),
                "normal": bool(p >= alpha),
                "skewness": safe_float(values.skew()),
            }
        )
    return out


def group_summary(df: pd.DataFrame, by: str, value: str) -> list[dict]:
    """Mean / median / count of a numeric column per category (used for driver charts)."""
    frame = df[[by, value]].dropna()
    grouped = frame.groupby(frame[by].astype(str))[value].agg(["count", "mean", "median"])
    grouped = grouped.sort_values("count", ascending=False).head(MAX_GROUPS)
    return [
        {"group": g, "count": int(r["count"]), "mean": safe_float(r["mean"]), "median": safe_float(r["median"])}
        for g, r in grouped.iterrows()
    ]
