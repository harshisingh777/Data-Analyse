"""Exploratory data analysis: column profiles, correlations, outliers and data quality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings
from .utils import pct, safe_float

# Semantic types, in the order they appear in the UI.
NUMERIC = "numeric"
CATEGORICAL = "categorical"
BOOLEAN = "boolean"
DATETIME = "datetime"
TEXT = "text"
IDENTIFIER = "identifier"
CONSTANT = "constant"
EMPTY = "empty"

SEMANTIC_TYPES = (NUMERIC, CATEGORICAL, BOOLEAN, DATETIME, TEXT, IDENTIFIER, CONSTANT, EMPTY)
MODELABLE_TYPES = (NUMERIC, CATEGORICAL, BOOLEAN, DATETIME)

_BOOL_TOKENS = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}
_ID_HINTS = ("id", "uuid", "guid", "key", "code", "number", "no")


def _looks_like_id_name(name: str) -> bool:
    lowered = name.lower().replace("-", "_").replace(" ", "_")
    parts = lowered.split("_")
    return lowered in ("id", "index") or parts[-1] in _ID_HINTS or lowered.endswith("id")


def infer_semantic_type(series: pd.Series) -> str:
    """Classify a column by how an analyst would treat it, not just its storage dtype."""
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return EMPTY
    nunique = non_null.nunique()
    if nunique == 1:
        return CONSTANT
    if pd.api.types.is_bool_dtype(series):
        return BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    if nunique == 2 and {str(v).strip().lower().removesuffix(".0") for v in non_null.unique()} <= _BOOL_TOKENS:
        return BOOLEAN
    # "Nearly unique" rather than strictly unique, so duplicated rows don't hide an ID column.
    nearly_unique = n > 20 and nunique / n >= 0.95
    if pd.api.types.is_numeric_dtype(series):
        is_integer = pd.api.types.is_integer_dtype(series) or bool((non_null % 1 == 0).all())
        if is_integer and nearly_unique and (_looks_like_id_name(str(series.name)) or non_null.is_monotonic_increasing):
            return IDENTIFIER
        return NUMERIC
    as_str = non_null.astype(str)
    if nearly_unique:
        return IDENTIFIER if as_str.str.len().mean() < 40 else TEXT
    if as_str.str.len().mean() > 60 or (nunique / n > 0.5 and n > 50):
        return TEXT
    return CATEGORICAL


def _histogram(values: pd.Series, bins: int = 20) -> dict:
    values = values.astype(float)
    if values.empty:
        return {"edges": [], "counts": []}
    bins = int(min(bins, max(5, np.sqrt(len(values)))))
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def iqr_outliers(values: pd.Series, k: float = 1.5) -> dict:
    """Tukey's fences: points beyond Q1 - k*IQR or Q3 + k*IQR."""
    values = values.dropna().astype(float)
    if values.empty:
        return {"count": 0, "pct": 0.0, "lower": None, "upper": None}
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    count = int(((values < lower) | (values > upper)).sum())
    return {"count": count, "pct": pct(count, len(values)), "lower": float(lower), "upper": float(upper)}


def _numeric_stats(values: pd.Series) -> dict:
    v = values.dropna().astype(float)
    quantiles = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    mean = v.mean()
    std = v.std()
    return {
        "mean": safe_float(mean),
        "std": safe_float(std),
        "min": safe_float(v.min()),
        "p5": safe_float(quantiles[0.05]),
        "q1": safe_float(quantiles[0.25]),
        "median": safe_float(quantiles[0.5]),
        "q3": safe_float(quantiles[0.75]),
        "p95": safe_float(quantiles[0.95]),
        "max": safe_float(v.max()),
        "iqr": safe_float(quantiles[0.75] - quantiles[0.25]),
        "sum": safe_float(v.sum()),
        "skewness": safe_float(v.skew()) if len(v) > 2 else None,
        "kurtosis": safe_float(v.kurt()) if len(v) > 3 else None,
        "cv": safe_float(std / mean) if mean else None,
        "zeros": int((v == 0).sum()),
        "zeros_pct": pct(int((v == 0).sum()), len(v)),
        "negatives": int((v < 0).sum()),
    }


def _top_values(series: pd.Series, limit: int = 10) -> list[dict]:
    non_null = series.dropna()
    counts = non_null.astype(str).value_counts()
    total = len(non_null)
    top = [{"value": str(k), "count": int(c), "pct": pct(c, total)} for k, c in counts.head(limit).items()]
    remainder = int(counts.iloc[limit:].sum())
    if remainder:
        top.append({"value": "(other)", "count": remainder, "pct": pct(remainder, total), "other": True})
    return top


def profile_column(series: pd.Series) -> dict:
    n = len(series)
    non_null = series.dropna()
    semantic = infer_semantic_type(series)
    nunique = int(non_null.nunique())
    profile = {
        "name": str(series.name),
        "dtype": str(series.dtype),
        "semantic_type": semantic,
        "count": int(len(non_null)),
        "missing": int(n - len(non_null)),
        "missing_pct": pct(n - len(non_null), n),
        "unique": nunique,
        "unique_pct": pct(nunique, len(non_null)),
        "sample_values": [str(v) for v in non_null.unique()[:5]],
    }
    if semantic in (NUMERIC, IDENTIFIER) and pd.api.types.is_numeric_dtype(series):
        profile["numeric"] = _numeric_stats(non_null)
        if semantic == NUMERIC:
            profile["outliers"] = iqr_outliers(non_null)
            profile["histogram"] = _histogram(non_null)
    if semantic == DATETIME:
        dt = pd.to_datetime(non_null)
        profile["datetime"] = {
            "min": dt.min().isoformat(),
            "max": dt.max().isoformat(),
            "range_days": int((dt.max() - dt.min()).days),
        }
        by_month = dt.dt.to_period("M").value_counts().sort_index()
        profile["timeline"] = [{"period": str(p), "count": int(c)} for p, c in by_month.tail(36).items()]
    if semantic in (CATEGORICAL, BOOLEAN, TEXT, CONSTANT, IDENTIFIER) and "numeric" not in profile:
        profile["top_values"] = _top_values(non_null)
    if semantic in (TEXT, CATEGORICAL, IDENTIFIER) and not pd.api.types.is_numeric_dtype(series):
        lengths = non_null.astype(str).str.len()
        profile["text"] = {
            "avg_length": safe_float(lengths.mean(), 2),
            "min_length": int(lengths.min()) if len(lengths) else 0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
        }
    return profile


def correlation_matrices(df: pd.DataFrame, columns: list[str]) -> dict:
    """Pearson and Spearman correlation matrices plus the strongest pairs."""
    columns = columns[: Settings.MAX_CORRELATION_COLUMNS]
    if len(columns) < 2:
        return {"columns": columns, "pearson": [], "spearman": [], "top_pairs": []}
    numeric = df[columns].astype(float)
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")

    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            r = pearson.loc[a, b]
            if pd.notna(r):
                pairs.append({"a": a, "b": b, "pearson": float(r), "spearman": safe_float(spearman.loc[a, b])})
    pairs.sort(key=lambda p: abs(p["pearson"]), reverse=True)

    def to_list(matrix: pd.DataFrame) -> list[list[float | None]]:
        return [[safe_float(v, 4) for v in row] for row in matrix.to_numpy()]

    return {
        "columns": columns,
        "pearson": to_list(pearson),
        "spearman": to_list(spearman),
        "top_pairs": pairs[:15],
    }


def quality_score(df: pd.DataFrame, columns: list[dict], duplicate_rows: int) -> dict:
    """A 0-100 data-quality score built from four weighted dimensions."""
    rows, cols = df.shape
    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum())
    numeric_cells = sum(c["count"] for c in columns if "outliers" in c)
    outlier_cells = sum(c["outliers"]["count"] for c in columns if "outliers" in c)
    problem_columns = sum(1 for c in columns if c["semantic_type"] in (CONSTANT, EMPTY) or c["missing_pct"] > 50)

    components = {
        "completeness": 1 - missing_cells / total_cells if total_cells else 1.0,
        "uniqueness": 1 - duplicate_rows / rows if rows else 1.0,
        "validity": 1 - outlier_cells / numeric_cells if numeric_cells else 1.0,
        "consistency": 1 - problem_columns / cols if cols else 1.0,
    }
    weights = {"completeness": 0.4, "uniqueness": 0.2, "validity": 0.2, "consistency": 0.2}
    score = 100 * sum(components[k] * w for k, w in weights.items())
    grade = next(g for threshold, g in ((90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")) if score >= threshold)
    return {
        "score": round(score, 1),
        "grade": grade,
        "components": {k: round(100 * v, 1) for k, v in components.items()},
        "weights": weights,
    }


def build_alerts(columns: list[dict], correlations: dict, duplicate_rows: int, rows: int) -> list[dict]:
    alerts = []

    def add(severity: str, kind: str, message: str, column: str | None = None):
        alerts.append({"severity": severity, "kind": kind, "column": column, "message": message})

    if duplicate_rows:
        add(
            "warning",
            "duplicates",
            f"{duplicate_rows:,} duplicate rows ({pct(duplicate_rows, rows):.1f}% of the data).",
        )
    for c in columns:
        name, semantic = c["name"], c["semantic_type"]
        if semantic == EMPTY:
            add("critical", "empty", f"'{name}' has no values.", name)
            continue
        if c["missing_pct"] >= 50:
            add("critical", "missing", f"'{name}' is {c['missing_pct']:.1f}% missing.", name)
        elif c["missing_pct"] >= 5:
            add("warning", "missing", f"'{name}' is {c['missing_pct']:.1f}% missing.", name)
        if semantic == CONSTANT:
            add("warning", "constant", f"'{name}' holds a single value and carries no information.", name)
        if semantic == IDENTIFIER:
            add("info", "identifier", f"'{name}' looks like an identifier and is excluded from modelling.", name)
        if semantic == TEXT:
            add("info", "text", f"'{name}' is free text / high cardinality and is excluded from modelling.", name)
        if semantic == CATEGORICAL and c["unique"] > 50:
            add("warning", "cardinality", f"'{name}' has {c['unique']:,} distinct categories.", name)
        if semantic in (CATEGORICAL, BOOLEAN) and c.get("top_values"):
            top = c["top_values"][0]
            if top["pct"] >= 90 and not top.get("other"):
                add("warning", "imbalance", f"'{name}' is dominated by '{top['value']}' ({top['pct']:.1f}%).", name)
        numeric = c.get("numeric") or {}
        skew = numeric.get("skewness")
        if semantic == NUMERIC and skew is not None and abs(skew) >= 2:
            direction = "right" if skew > 0 else "left"
            add("info", "skew", f"'{name}' is highly {direction}-skewed (skewness {skew:.2f}).", name)
        outliers = c.get("outliers")
        if outliers and outliers["pct"] >= 5:
            add(
                "warning",
                "outliers",
                f"'{name}' has {outliers['count']:,} IQR outliers ({outliers['pct']:.1f}%).",
                name,
            )
    for pair in correlations.get("top_pairs", []):
        if abs(pair["pearson"]) >= 0.9:
            add(
                "warning",
                "correlation",
                f"'{pair['a']}' and '{pair['b']}' are almost collinear (r = {pair['pearson']:.2f}).",
            )
    order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: order[a["severity"]])
    return alerts


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Full exploratory profile of a dataset."""
    rows, cols = df.shape
    columns = [profile_column(df[c]) for c in df.columns]
    semantic = {c["name"]: c["semantic_type"] for c in columns}

    try:
        duplicate_rows = int(df.duplicated().sum())
    except TypeError:  # unhashable values (e.g. nested JSON)
        duplicate_rows = int(df.astype(str).duplicated().sum())
    missing_cells = int(df.isna().sum().sum())

    numeric_cols = [c for c in df.columns if semantic[c] == NUMERIC]
    correlations = correlation_matrices(df, numeric_cols)
    type_counts = {t: sum(1 for v in semantic.values() if v == t) for t in SEMANTIC_TYPES}

    return {
        "overview": {
            "rows": rows,
            "columns": cols,
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "missing_cells": missing_cells,
            "missing_pct": pct(missing_cells, rows * cols),
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": pct(duplicate_rows, rows),
            "type_counts": {k: v for k, v in type_counts.items() if v},
        },
        "columns": columns,
        "correlations": correlations,
        "missing": sorted(
            ({"column": c["name"], "count": c["missing"], "pct": c["missing_pct"]} for c in columns if c["missing"]),
            key=lambda m: m["count"],
            reverse=True,
        ),
        "quality": quality_score(df, columns, duplicate_rows),
        "alerts": build_alerts(columns, correlations, duplicate_rows, rows),
    }


def semantic_types(profile: dict) -> dict[str, str]:
    return {c["name"]: c["semantic_type"] for c in profile["columns"]}
