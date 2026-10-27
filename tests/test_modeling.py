import io

import joblib
import numpy as np
import pandas as pd
import pytest

from datalens.modeling import DateFeatures, TrainConfig, plan_features, resolve_model_keys, train_models
from datalens.profiling import profile_dataframe, semantic_types
from datalens.utils import DataLensError

FAST = ["linear", "decision_tree"]


def _semantic(df):
    return semantic_types(profile_dataframe(df))


def test_feature_plan_excludes_identifiers(classification_df):
    plan = plan_features(classification_df, "churn", _semantic(classification_df))
    assert plan.numeric == ["tenure", "tickets", "noise"]
    assert plan.categorical == ["contract"]
    assert plan.datetime == ["signup"]
    assert plan.excluded == {"customer_id": "identifier"}


def test_resolve_model_keys_maps_linear_alias():
    assert resolve_model_keys(["linear", "knn", "logistic_regression"], "classification") == [
        "logistic_regression",
        "knn",
    ]
    assert resolve_model_keys(["linear"], "regression") == ["linear_regression"]


def test_date_features_expand_and_name_columns():
    X = pd.DataFrame({"d": pd.to_datetime(["2024-03-15", None])})
    out = DateFeatures().fit(X).transform(X)
    assert list(out.columns) == ["d_year", "d_month", "d_dayofweek", "d_dayofyear"]
    assert out.iloc[0].tolist() == [2024, 3, 4, 75]
    assert out.iloc[1].isna().all()


def test_classification_run_beats_baseline(classification_df):
    result = train_models(
        classification_df, TrainConfig(target="churn", models=FAST, cv_folds=3), _semantic(classification_df)
    )
    s = result.summary
    assert s["task"] == "classification"
    assert [r["key"] for r in s["leaderboard"]][-1] == "baseline"
    assert {r["key"] for r in s["leaderboard"]} == {"baseline", "logistic_regression", "decision_tree"}
    assert s["leaderboard"][0]["rank"] == 1
    ev = s["evaluation"]["metrics"]
    assert ev["accuracy"] > ev["baseline_accuracy"]
    assert 0.5 < ev["roc_auc"] <= 1
    cm = s["evaluation"]["confusion_matrix"]
    assert sum(map(sum, cm["matrix"])) == s["test_rows"]
    assert s["evaluation"]["roc_curve"]["positive_label"] == "Yes"
    features = [f["feature"] for f in s["feature_importance"]]
    assert set(features) == {"tenure", "tickets", "noise", "contract", "signup"}
    assert features.index("tickets") < features.index("noise")


def test_regression_run_and_prediction(regression_df):
    result = train_models(regression_df, TrainConfig(target="price", models=FAST, cv_folds=3), _semantic(regression_df))
    s = result.summary
    assert s["task"] == "regression"
    assert s["evaluation"]["metrics"]["r2"] > 0.8
    assert s["evaluation"]["metrics"]["mae"] < s["evaluation"]["metrics"]["baseline_mae"]
    assert len(s["evaluation"]["scatter"]) == s["test_rows"]
    preds = result.predict([{"size": 100, "rooms": 3, "city": "C"}, {"size": "80", "city": "unseen"}])
    assert len(preds) == 2
    assert preds[0]["prediction"] > preds[1]["prediction"]


def test_classification_prediction_returns_probabilities(classification_df):
    result = train_models(
        classification_df, TrainConfig(target="churn", models=["linear"], cv_folds=3), _semantic(classification_df)
    )
    [pred] = result.predict([{"tenure": 2, "tickets": 6, "contract": "monthly", "signup": "2023-01-01"}])
    assert pred["prediction"] in ("Yes", "No")
    assert sum(pred["probabilities"].values()) == pytest.approx(1.0)


def test_tuning_reports_scores(regression_df):
    result = train_models(
        regression_df,
        TrainConfig(target="price", models=["decision_tree"], cv_folds=3, tune=True, tune_iterations=4),
        _semantic(regression_df),
    )
    tuning = result.summary["tuning"]
    assert tuning["iterations"] == 4
    assert "max_depth" in tuning["best_params"]
    assert "estimator" not in tuning


def test_exported_model_round_trips(regression_df):
    result = train_models(
        regression_df, TrainConfig(target="price", models=["linear"], cv_folds=3), _semantic(regression_df)
    )
    bundle = joblib.load(io.BytesIO(result.to_bytes()))
    assert bundle["target"] == "price"
    restored = bundle["pipeline"].predict(regression_df[bundle["features"]].head(5))
    np.testing.assert_allclose(restored, result.pipeline.predict(regression_df[bundle["features"]].head(5)))


def test_rare_classes_are_dropped_with_a_note(classification_df):
    df = classification_df.copy()
    df.loc[0, "churn"] = "Maybe"
    result = train_models(df, TrainConfig(target="churn", models=["linear"], cv_folds=3), _semantic(df))
    assert any("single example" in n for n in result.summary["notes"])
    assert "Maybe" not in result.summary["evaluation"]["confusion_matrix"]["labels"]


@pytest.mark.parametrize(
    "config,message",
    [
        (TrainConfig(target="nope"), "does not exist"),
        (TrainConfig(target="price", test_size=0.9), "test_size"),
        (TrainConfig(target="price", cv_folds=1), "cv_folds"),
        (TrainConfig(target="price", models=["svm"]), "Unknown model"),
        (TrainConfig(target="city", task="regression"), "numeric target"),
    ],
)
def test_invalid_configs_raise(regression_df, config, message):
    with pytest.raises(DataLensError, match=message):
        train_models(regression_df, config, _semantic(regression_df))


def test_too_few_rows_raises():
    df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
    with pytest.raises(DataLensError, match="At least 20 rows"):
        train_models(df, TrainConfig(target="y"), _semantic(df))
