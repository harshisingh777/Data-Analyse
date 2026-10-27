"""Choosing a target column and deciding between classification and regression."""

from __future__ import annotations

import pandas as pd

from .profiling import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, infer_semantic_type
from .utils import DataLensError, pct

CLASSIFICATION = "classification"
REGRESSION = "regression"

# Outcome-style names come before amount-style names: in a sales table "returned" is a more
# likely prediction target than "revenue", which is often derived from other columns.
TARGET_HINTS = (
    "target",
    "label",
    "y",
    "class",
    "outcome",
    "churn",
    "churned",
    "survived",
    "default",
    "fraud",
    "returned",
    "converted",
    "result",
    "status",
    "price",
    "salary",
    "sales",
    "revenue",
)
MAX_CLASSES = 20


def infer_target(df: pd.DataFrame, semantic: dict[str, str] | None = None) -> str | None:
    """Pick the most plausible prediction target.

    Preference order: a column named like a well-known target, then the last
    usable column (the common convention for ML datasets), then any low-cardinality
    categorical column.
    """
    semantic = semantic or {c: infer_semantic_type(df[c]) for c in df.columns}
    usable = [c for c in df.columns if semantic.get(c) in (NUMERIC, CATEGORICAL, BOOLEAN)]
    if not usable:
        return None
    lowered = {c.lower().replace(" ", "_"): c for c in usable}
    for hint in TARGET_HINTS:
        if hint in lowered:
            return lowered[hint]
    last = df.columns[-1]
    if last in usable:
        return last
    for c in usable:
        if semantic[c] in (CATEGORICAL, BOOLEAN) and df[c].nunique() <= MAX_CLASSES:
            return c
    return usable[-1]


def infer_task(series: pd.Series) -> str:
    """Classification for labels and small integer sets, regression for continuous values."""
    non_null = series.dropna()
    if non_null.empty:
        raise DataLensError(f"Target '{series.name}' has no values.")
    semantic = infer_semantic_type(series)
    if semantic == DATETIME:
        raise DataLensError(f"Target '{series.name}' is a date column and cannot be predicted directly.")
    if not pd.api.types.is_numeric_dtype(series) or semantic == BOOLEAN:
        return CLASSIFICATION
    nunique = non_null.nunique()
    is_integer = bool((non_null.astype(float) % 1 == 0).all())
    if is_integer and nunique <= 10:
        return CLASSIFICATION
    return REGRESSION


def class_balance(series: pd.Series) -> list[dict]:
    counts = series.dropna().astype(str).value_counts()
    total = int(counts.sum())
    return [{"label": k, "count": int(v), "pct": pct(v, total)} for k, v in counts.items()]


def validate_target(df: pd.DataFrame, target: str, task: str) -> None:
    if target not in df.columns:
        raise DataLensError(f"Column '{target}' does not exist.")
    if task not in (CLASSIFICATION, REGRESSION):
        raise DataLensError(f"Unknown task '{task}'. Use 'classification' or 'regression'.")
    series = df[target].dropna()
    if series.nunique() < 2:
        raise DataLensError(f"Target '{target}' needs at least two distinct values.")
    if task == CLASSIFICATION and series.nunique() > 50:
        raise DataLensError(
            f"Target '{target}' has {series.nunique()} classes; treat it as regression or choose another column."
        )
    if task == REGRESSION and not pd.api.types.is_numeric_dtype(df[target]):
        raise DataLensError(f"Regression needs a numeric target, but '{target}' is not numeric.")
