import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from datalens.profiling import profile_dataframe, semantic_types
from datalens.stats import association_tests, benjamini_hochberg, effect_label, group_summary, normality_tests
from datalens.target import CLASSIFICATION, REGRESSION, infer_target, infer_task, validate_target
from datalens.utils import DataLensError


def test_infer_target_prefers_known_names_then_last_column():
    df = pd.DataFrame({"churned": ["y", "n"] * 10, "x": range(20), "z": range(20)})
    assert infer_target(df) == "churned"
    df2 = pd.DataFrame({"a": np.arange(20.0) / 3, "b": ["p", "q"] * 10})
    assert infer_target(df2) == "b"


def test_infer_target_skips_identifier_last_column():
    df = pd.DataFrame({"kind": ["a", "b"] * 30, "row_id": range(60)})
    assert infer_target(df) == "kind"


@pytest.mark.parametrize(
    "values,expected",
    [
        (["a", "b", "a"], CLASSIFICATION),
        ([0, 1, 1, 0], CLASSIFICATION),
        ([1, 2, 3, 4, 5, 1, 2], CLASSIFICATION),
        (list(np.linspace(0, 10, 50)), REGRESSION),
        (list(range(100)), REGRESSION),
    ],
)
def test_infer_task(values, expected):
    assert infer_task(pd.Series(values)) == expected


def test_validate_target_rejects_bad_choices():
    df = pd.DataFrame({"t": ["a"] * 10, "x": range(10), "s": list("abcdefghij")})
    with pytest.raises(DataLensError, match="two distinct"):
        validate_target(df, "t", CLASSIFICATION)
    with pytest.raises(DataLensError, match="does not exist"):
        validate_target(df, "missing", CLASSIFICATION)
    with pytest.raises(DataLensError, match="numeric target"):
        validate_target(df, "s", REGRESSION)


def test_benjamini_hochberg_matches_reference_values():
    p = [0.01, 0.04, 0.03, 0.005, None]
    adjusted = benjamini_hochberg(p)
    # Reference: statsmodels.multipletests(method="fdr_bh") on the four finite values.
    assert adjusted[:4] == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert adjusted[4] is None


def test_effect_labels():
    assert effect_label(0.05, "r") == "negligible"
    assert effect_label(-0.35, "r") == "medium"
    assert effect_label(0.2, "eta_squared") == "large"
    assert effect_label(None, "r") == "n/a"


def test_association_tests_find_real_signal(classification_df):
    profile = profile_dataframe(classification_df)
    result = association_tests(classification_df, "churn", CLASSIFICATION, semantic_types(profile))
    by_feature = {t["feature"]: t for t in result["tests"]}
    assert by_feature["tickets"]["significant"]
    assert by_feature["contract"]["test"] == "Chi-square independence"
    assert by_feature["tickets"]["test"] == "One-way ANOVA"
    assert not by_feature["noise"]["significant"]
    assert "customer_id" not in by_feature  # identifiers are not tested


def test_anova_matches_scipy(regression_df):
    profile = profile_dataframe(regression_df)
    result = association_tests(regression_df, "price", REGRESSION, semantic_types(profile))
    city = next(t for t in result["tests"] if t["feature"] == "city")
    groups = [g["price"].to_numpy() for _, g in regression_df.groupby("city")]
    f_stat, p = sps.f_oneway(*groups)
    assert city["statistic"] == pytest.approx(f_stat)
    assert city["p_value"] == pytest.approx(p)
    size = next(t for t in result["tests"] if t["feature"] == "size")
    assert size["statistic"] == pytest.approx(regression_df["size"].corr(regression_df["price"]))


def test_normality_tests_distinguish_distributions():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"normal": rng.normal(size=2000), "skewed": rng.exponential(size=2000)})
    result = {r["column"]: r for r in normality_tests(df, ["normal", "skewed"])}
    assert result["normal"]["normal"]
    assert not result["skewed"]["normal"]


def test_group_summary(regression_df):
    rows = group_summary(regression_df, "city", "price")
    assert {r["group"] for r in rows} == {"A", "B", "C"}
    assert sum(r["count"] for r in rows) == len(regression_df)
