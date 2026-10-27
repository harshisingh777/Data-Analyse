import numpy as np
import pandas as pd
import pytest

from datalens.profiling import (
    BOOLEAN,
    CATEGORICAL,
    CONSTANT,
    DATETIME,
    EMPTY,
    IDENTIFIER,
    NUMERIC,
    TEXT,
    infer_semantic_type,
    iqr_outliers,
    profile_dataframe,
    quality_score,
)


@pytest.mark.parametrize(
    "series,expected",
    [
        (pd.Series([None, None], dtype=object), EMPTY),
        (pd.Series([5, 5, 5]), CONSTANT),
        (pd.Series(["yes", "no", "yes"]), BOOLEAN),
        (pd.Series([0, 1, 1, 0]), BOOLEAN),
        (pd.Series(pd.date_range("2024-01-01", periods=5)), DATETIME),
        (pd.Series([1.5, 2.25, 3.0, 4.75]), NUMERIC),
        (pd.Series(["a", "b", "a", "c"] * 10), CATEGORICAL),
        (pd.Series(range(100), name="order_id"), IDENTIFIER),
        (pd.Series([f"SKU-{i}" for i in range(50)]), IDENTIFIER),
        (pd.Series([f"a long free-text comment number {i} with plenty of words in it" for i in range(60)]), TEXT),
    ],
)
def test_semantic_type_inference(series, expected):
    assert infer_semantic_type(series) == expected


def test_nearly_unique_ids_survive_duplicate_rows():
    ids = pd.Series([f"C{i}" for i in range(100)] + ["C1", "C2"], name="customer")
    assert infer_semantic_type(ids) == IDENTIFIER


def test_iqr_outliers_matches_tukey_fences():
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100])
    result = iqr_outliers(values)
    assert result["count"] == 1
    assert result["upper"] == pytest.approx(7 + 1.5 * 4)


def test_profile_shape_and_overview(classification_df):
    profile = profile_dataframe(classification_df)
    ov = profile["overview"]
    assert ov["rows"] == 300 and ov["columns"] == 7
    assert ov["missing_cells"] == classification_df.isna().sum().sum()
    names = [c["name"] for c in profile["columns"]]
    assert names == list(classification_df.columns)
    tenure = next(c for c in profile["columns"] if c["name"] == "tenure")
    assert tenure["numeric"]["median"] == pytest.approx(classification_df["tenure"].median())
    assert sum(tenure["histogram"]["counts"]) == tenure["count"]


def test_correlations_and_alerts_detect_collinearity():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "x_copy": x * 2 + 0.01 * rng.normal(size=200), "z": rng.normal(size=200)})
    profile = profile_dataframe(df)
    top = profile["correlations"]["top_pairs"][0]
    assert {top["a"], top["b"]} == {"x", "x_copy"}
    assert top["pearson"] > 0.99
    assert any(a["kind"] == "correlation" for a in profile["alerts"])
    matrix = profile["correlations"]["pearson"]
    assert matrix[0][0] == pytest.approx(1.0)


def test_alerts_cover_missing_constant_and_duplicates():
    df = pd.DataFrame(
        {
            "a": [1, 2, None, None, None, 6] * 5,
            "const": ["x"] * 30,
            "b": [0, 1, 2] * 10,  # repeats every 6 rows alongside "a", creating duplicates
        }
    )
    profile = profile_dataframe(df)
    kinds = {a["kind"] for a in profile["alerts"]}
    assert {"missing", "constant", "duplicates"} <= kinds
    assert profile["alerts"][0]["severity"] in ("critical", "warning")


def test_quality_score_is_perfect_for_clean_data_and_bounded():
    clean = pd.DataFrame({"a": range(50), "b": list("xy") * 25, "c": np.linspace(0, 1, 50)})
    profile = profile_dataframe(clean)
    assert profile["quality"]["score"] == pytest.approx(100.0)
    assert profile["quality"]["grade"] == "A"

    dirty = pd.DataFrame({"a": [None] * 10 + [1] * 10})
    columns = profile_dataframe(dirty)["columns"]
    q = quality_score(dirty, columns, duplicate_rows=9)
    assert 0 <= q["score"] < 70
