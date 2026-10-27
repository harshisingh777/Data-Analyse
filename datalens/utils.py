"""Small shared helpers."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


class DataLensError(Exception):
    """An expected, user-facing error (bad input, impossible request)."""

    status_code = 400

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas values into plain JSON-safe Python types.

    NaN and infinity become ``None`` so the output is valid strict JSON.
    """
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def safe_float(value: Any, digits: int | None = None) -> float | None:
    """Convert to a finite float (optionally rounded), or ``None``."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return round(result, digits) if digits is not None else result


def pct(part: float, whole: float) -> float:
    """Percentage of ``part`` in ``whole`` (0 when ``whole`` is 0)."""
    return 0.0 if not whole else 100.0 * part / whole
