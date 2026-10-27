"""Loading raw files into clean, well-typed DataFrames."""

from __future__ import annotations

import csv
import io
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DataLensError

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json", ".parquet")

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", "#n/a"}

_NUMERIC_JUNK = re.compile(r"[,\s$€£₹%]")
_DATE_LIKE = re.compile(
    r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})([ T]\d{1,2}:\d{2}(:\d{2})?)?"
)


def load_dataframe(data: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded bytes into a cleaned DataFrame based on the file extension."""
    if not data:
        raise DataLensError("The uploaded file is empty.")
    ext = Path(filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(SUPPORTED_EXTENSIONS)
        raise DataLensError(f"Unsupported file type '{ext or 'unknown'}'. Supported types: {allowed}.")

    try:
        if ext in (".csv", ".tsv", ".txt"):
            df = _read_delimited(data, ext)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(data))
        elif ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(data))
        else:
            df = _read_json(data)
    except DataLensError:
        raise
    except Exception as exc:  # parser errors vary widely between formats
        raise DataLensError(f"Could not parse '{filename}': {exc}") from exc

    return prepare_dataframe(df)


def load_path(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataLensError(f"File not found: {path}")
    return load_dataframe(path.read_bytes(), path.name)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DataLensError("Could not decode the file. Please save it as UTF-8.")


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_delimited(data: bytes, ext: str) -> pd.DataFrame:
    text = _decode(data)
    sep = "\t" if ext == ".tsv" else _sniff_delimiter(text)
    return pd.read_csv(io.StringIO(text), sep=sep, low_memory=False, skipinitialspace=True)


def _read_json(data: bytes) -> pd.DataFrame:
    payload = json.loads(_decode(data))
    if isinstance(payload, dict):
        # {"data": [...]} or {"col": [...], ...}
        list_values = [v for v in payload.values() if isinstance(v, list)]
        if len(payload) == 1 and list_values:
            payload = list_values[0]
        else:
            return pd.DataFrame(payload)
    if not isinstance(payload, list):
        raise DataLensError("JSON must be an array of records or an object of columns.")
    return pd.json_normalize(payload)


def _clean_column_names(columns) -> list[str]:
    seen: dict[str, int] = {}
    names = []
    for i, col in enumerate(columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        names.append(name)
    return names


def _try_numeric(series: pd.Series) -> pd.Series | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    stripped = non_null.astype(str).str.replace(_NUMERIC_JUNK, "", regex=True)
    stripped = stripped.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # accounting negatives
    converted = pd.to_numeric(stripped, errors="coerce")
    if converted.notna().mean() < 0.95:
        return None
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    result.loc[converted.index] = converted
    if result.dropna().apply(float.is_integer).all():
        return result.astype("Int64") if result.isna().any() else result.astype("int64")
    return result


def _try_datetime(series: pd.Series) -> pd.Series | None:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    sample = non_null.head(200)
    if sample.str.match(_DATE_LIKE).mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
        if parsed.notna().sum() < 0.9 * len(non_null):
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.notna().sum() < 0.9 * len(non_null):
        return None
    return parsed


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, null tokens and infer numeric / datetime columns."""
    if df is None or df.shape[1] == 0:
        raise DataLensError("The file contains no columns.")
    df = df.copy()
    df.columns = _clean_column_names(df.columns)
    df = df.dropna(how="all")
    if df.empty:
        raise DataLensError("The file contains no data rows.")
    df = df.reset_index(drop=True)

    for col in df.columns:
        series = df[col]
        if series.dtype != object and not pd.api.types.is_string_dtype(series):
            continue
        series = series.map(lambda v: v.strip() if isinstance(v, str) else v)
        series = series.map(lambda v: np.nan if isinstance(v, str) and v.lower() in NULL_TOKENS else v)
        if series.dropna().map(lambda v: isinstance(v, bool)).all() and series.notna().any():
            df[col] = series.astype("boolean")
            continue
        numeric = _try_numeric(series)
        if numeric is not None:
            df[col] = numeric
            continue
        parsed = _try_datetime(series)
        df[col] = parsed if parsed is not None else series.astype(object)
    return df
