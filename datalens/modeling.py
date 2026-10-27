"""AutoML: preprocessing, model comparison with cross-validation, tuning and evaluation."""

from __future__ import annotations

import io
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import Settings
from .profiling import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC
from .target import CLASSIFICATION, REGRESSION, validate_target
from .utils import DataLensError, safe_float

# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    factory: Callable[[int], BaseEstimator]
    search_space: dict = field(default_factory=dict)


CLASSIFIERS = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "logistic_regression",
            "Logistic Regression",
            lambda rs: LogisticRegression(max_iter=2000),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        ModelSpec(
            "decision_tree",
            "Decision Tree",
            lambda rs: DecisionTreeClassifier(max_depth=8, random_state=rs),
            {"model__max_depth": randint(2, 20), "model__min_samples_leaf": randint(1, 20)},
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            lambda rs: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=rs),
            {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", 0.5],
            },
        ),
        ModelSpec(
            "gradient_boosting",
            "Gradient Boosting",
            lambda rs: HistGradientBoostingClassifier(random_state=rs),
            {
                "model__learning_rate": loguniform(0.02, 0.3),
                "model__max_leaf_nodes": randint(8, 64),
                "model__l2_regularization": loguniform(1e-4, 1.0),
            },
        ),
        ModelSpec(
            "knn",
            "k-Nearest Neighbours",
            lambda rs: KNeighborsClassifier(n_neighbors=15),
            {"model__n_neighbors": randint(3, 50), "model__weights": ["uniform", "distance"]},
        ),
    )
}

REGRESSORS = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "linear_regression",
            "Ridge Regression",
            lambda rs: Ridge(alpha=1.0),
            {"model__alpha": loguniform(1e-3, 1e2)},
        ),
        ModelSpec(
            "decision_tree",
            "Decision Tree",
            lambda rs: DecisionTreeRegressor(max_depth=8, random_state=rs),
            {"model__max_depth": randint(2, 20), "model__min_samples_leaf": randint(1, 20)},
        ),
        ModelSpec(
            "random_forest",
            "Random Forest",
            lambda rs: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=rs),
            {
                "model__n_estimators": randint(100, 400),
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": [1.0, "sqrt", 0.5],
            },
        ),
        ModelSpec(
            "gradient_boosting",
            "Gradient Boosting",
            lambda rs: HistGradientBoostingRegressor(random_state=rs),
            {
                "model__learning_rate": loguniform(0.02, 0.3),
                "model__max_leaf_nodes": randint(8, 64),
                "model__l2_regularization": loguniform(1e-4, 1.0),
            },
        ),
        ModelSpec(
            "knn",
            "k-Nearest Neighbours",
            lambda rs: KNeighborsRegressor(n_neighbors=15),
            {"model__n_neighbors": randint(3, 50), "model__weights": ["uniform", "distance"]},
        ),
    )
}

# Metric definitions: scikit-learn scorer name, label, and whether higher is better.
CLASSIFICATION_SCORERS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_weighted": "f1_weighted",
}
REGRESSION_SCORERS = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}
PRIMARY_METRIC = {CLASSIFICATION: "f1_weighted", REGRESSION: "r2"}


LINEAR_ALIASES = ("linear", "linear_regression", "logistic_regression")


def model_catalogue(task: str) -> dict[str, ModelSpec]:
    return CLASSIFIERS if task == CLASSIFICATION else REGRESSORS


def resolve_model_keys(keys: list[str], task: str) -> list[str]:
    """Map task-agnostic names (e.g. 'linear') to the catalogue keys for ``task``, removing duplicates."""
    linear_key = "logistic_regression" if task == CLASSIFICATION else "linear_regression"
    resolved = []
    for key in keys:
        key = linear_key if key in LINEAR_ALIASES else key
        if key not in resolved:
            resolved.append(key)
    return resolved


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


class DateFeatures(BaseEstimator, TransformerMixin):
    """Expand datetime columns into year / month / day-of-week / day-of-year numbers."""

    PARTS = ("year", "month", "dayofweek", "dayofyear")

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        out = {}
        for col in self.columns_:
            parsed = pd.to_datetime(X[col], errors="coerce")
            for part in self.PARTS:
                out[f"{col}_{part}"] = getattr(parsed.dt, part).astype(float)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_{p}" for c in self.columns_ for p in self.PARTS], dtype=object)


def _to_string_categories(X):
    """Cast categorical inputs to strings so mixed types (bools, ints, text) encode consistently."""
    frame = pd.DataFrame(X)
    return frame.apply(lambda s: s.map(lambda v: np.nan if pd.isna(v) else str(v))).astype(object)


@dataclass
class FeaturePlan:
    numeric: list[str]
    categorical: list[str]
    datetime: list[str]
    excluded: dict[str, str]

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical + self.datetime


def plan_features(df: pd.DataFrame, target: str, semantic: dict[str, str]) -> FeaturePlan:
    numeric, categorical, dates, excluded = [], [], [], {}
    for col in df.columns:
        if col == target:
            continue
        kind = semantic.get(col)
        if kind == NUMERIC:
            numeric.append(col)
        elif kind in (CATEGORICAL, BOOLEAN):
            categorical.append(col)
        elif kind == DATETIME:
            dates.append(col)
        else:
            excluded[col] = kind or "unknown"
    if not (numeric or categorical or dates):
        raise DataLensError("No usable feature columns were found besides the target.")
    return FeaturePlan(numeric, categorical, dates, excluded)


def build_preprocessor(plan: FeaturePlan) -> ColumnTransformer:
    transformers = []
    if plan.numeric:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                plan.numeric,
            )
        )
    if plan.categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("as_str", FunctionTransformer(_to_string_categories, feature_names_out="one-to-one")),
                        ("impute", SimpleImputer(strategy="constant", fill_value="(missing)")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, sparse_output=False),
                        ),
                    ]
                ),
                plan.categorical,
            )
        )
    if plan.datetime:
        transformers.append(
            (
                "date",
                Pipeline(
                    [
                        ("expand", DateFeatures()),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                plan.datetime,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    target: str
    task: str | None = None
    models: list[str] | None = None
    test_size: float = 0.2
    cv_folds: int = 5
    tune: bool = False
    tune_iterations: int = 12
    random_state: int = Settings.RANDOM_STATE

    def validate(self) -> None:
        if not 0.1 <= self.test_size <= 0.5:
            raise DataLensError("test_size must be between 0.1 and 0.5.")
        if not 2 <= self.cv_folds <= 10:
            raise DataLensError("cv_folds must be between 2 and 10.")


@dataclass
class TrainResult:
    pipeline: Pipeline
    summary: dict
    plan: FeaturePlan
    task: str
    target: str

    def predict(self, records: list[dict]) -> list[dict]:
        if not records:
            raise DataLensError("Provide at least one record to predict.")
        frame = pd.DataFrame(records)
        for col in self.plan.features:
            if col not in frame.columns:
                frame[col] = np.nan
        for col in self.plan.numeric:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame[self.plan.features]
        predictions = self.pipeline.predict(frame)
        out = [{"prediction": p.item() if hasattr(p, "item") else p} for p in predictions]
        if self.task == CLASSIFICATION and hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(frame)
            classes = [str(c) for c in self.pipeline.classes_]
            for row, probs in zip(out, proba, strict=True):
                row["probabilities"] = dict(zip(classes, (float(p) for p in probs), strict=True))
        return out

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        joblib.dump(
            {"pipeline": self.pipeline, "features": self.plan.features, "target": self.target, "task": self.task},
            buffer,
        )
        return buffer.getvalue()


def _cv_splitter(task: str, y: pd.Series, folds: int, random_state: int):
    if task == CLASSIFICATION:
        smallest_class = int(y.value_counts().min())
        folds = max(2, min(folds, smallest_class))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


def _summarise_cv(scores: dict, names: dict[str, str]) -> dict:
    out = {}
    for key, scorer in names.items():
        values = np.asarray(scores[f"test_{key}"], dtype=float)
        if scorer.startswith("neg_"):
            values = -values
        out[key] = {"mean": safe_float(np.nanmean(values)), "std": safe_float(np.nanstd(values))}
    return out


def _prepare_xy(df: pd.DataFrame, config: TrainConfig, plan: FeaturePlan, warnings_out: list[str]):
    data = df[plan.features + [config.target]].dropna(subset=[config.target])
    dropped = len(df) - len(data)
    if dropped:
        warnings_out.append(f"Dropped {dropped:,} rows with a missing target value.")
    if len(data) > Settings.MAX_TRAIN_ROWS:
        data = data.sample(Settings.MAX_TRAIN_ROWS, random_state=config.random_state)
        warnings_out.append(f"Trained on a random sample of {Settings.MAX_TRAIN_ROWS:,} rows for speed.")
    y = data[config.target]
    if config.task == CLASSIFICATION:
        y = y.astype(str)
        counts = y.value_counts()
        rare = counts[counts < 2].index
        if len(rare):
            data = data[~y.isin(rare)]
            y = y[~y.isin(rare)]
            warnings_out.append(
                f"Removed {len(rare)} class(es) with a single example: {', '.join(map(str, rare[:5]))}."
            )
        if y.nunique() < 2:
            raise DataLensError("The target needs at least two classes with two or more rows each.")
    else:
        y = y.astype(float)
    if len(data) < 20:
        raise DataLensError("At least 20 rows with a target value are needed to train a model.")
    return data[plan.features], y


def train_models(df: pd.DataFrame, config: TrainConfig, semantic: dict[str, str]) -> TrainResult:
    """Compare candidate models with cross-validation, then evaluate the winner on a hold-out set."""
    from .target import infer_task  # local import keeps module import order simple

    config.validate()
    if config.target not in df.columns:
        raise DataLensError(f"Column '{config.target}' does not exist.")
    config.task = config.task or infer_task(df[config.target])
    validate_target(df, config.target, config.task)

    catalogue = model_catalogue(config.task)
    selected = resolve_model_keys(config.models, config.task) if config.models else list(catalogue)
    unknown = [m for m in selected if m not in catalogue]
    if unknown:
        raise DataLensError(f"Unknown model(s) for {config.task}: {', '.join(unknown)}.")

    notes: list[str] = []
    plan = plan_features(df, config.target, semantic)
    X, y = _prepare_xy(df, config, plan, notes)

    stratify = y if config.task == CLASSIFICATION else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state, stratify=stratify
        )
    except ValueError:  # too many classes for a stratified split of this size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        notes.append("The hold-out split could not be stratified; class proportions may differ slightly.")
    cv = _cv_splitter(config.task, y_train, config.cv_folds, config.random_state)
    scorers = CLASSIFICATION_SCORERS if config.task == CLASSIFICATION else REGRESSION_SCORERS
    primary = PRIMARY_METRIC[config.task]
    preprocessor = build_preprocessor(plan)

    # Baseline first, so every model can be judged against a naive predictor.
    baseline_est = DummyClassifier(strategy="most_frequent") if config.task == CLASSIFICATION else DummyRegressor()
    candidates = [("baseline", "Baseline (naive)", baseline_est)]
    candidates += [(key, catalogue[key].name, catalogue[key].factory(config.random_state)) for key in selected]

    leaderboard = []
    for key, name, estimator in candidates:
        pipe = Pipeline([("pre", clone(preprocessor)), ("model", estimator)])
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipe, X_train, y_train, cv=cv, scoring=scorers, error_score=np.nan, return_train_score=True
            )
        metrics = _summarise_cv(scores, scorers)
        train_primary = np.asarray(scores[f"train_{primary}"], dtype=float)
        leaderboard.append(
            {
                "key": key,
                "name": name,
                "baseline": key == "baseline",
                "cv": metrics,
                "train_score": safe_float(np.nanmean(train_primary)),
                "fit_seconds": round(time.perf_counter() - start, 2),
            }
        )

    ranked = sorted(
        (row for row in leaderboard if not row["baseline"]),
        key=lambda r: -(r["cv"][primary]["mean"] if r["cv"][primary]["mean"] is not None else -np.inf),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    leaderboard.sort(key=lambda r: r.get("rank", 0) if not r["baseline"] else 999)
    best = ranked[0]
    best_pipe = Pipeline([("pre", clone(preprocessor)), ("model", catalogue[best["key"]].factory(config.random_state))])

    tuning = None
    if config.tune and catalogue[best["key"]].search_space:
        tuning = _tune(
            best_pipe,
            catalogue[best["key"]],
            X_train,
            y_train,
            cv,
            scorers[primary],
            best["cv"][primary]["mean"],
            config,
        )
        if tuning["improved"]:
            best_pipe = tuning.pop("estimator")
        else:
            tuning.pop("estimator")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_pipe.fit(X_train, y_train)

    evaluation = (
        _evaluate_classifier(best_pipe, X_train, y_train, X_test, y_test)
        if config.task == CLASSIFICATION
        else _evaluate_regressor(best_pipe, X_train, y_train, X_test, y_test)
    )
    importances = _permutation_importance(best_pipe, X_test, y_test, scorers[primary], config.random_state)

    train_score = best["train_score"]
    cv_score = best["cv"][primary]["mean"]
    overfit_gap = (train_score - cv_score) if train_score is not None and cv_score is not None else None

    summary = {
        "target": config.target,
        "task": config.task,
        "primary_metric": primary,
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "cv_folds": cv.get_n_splits(),
        "features": {
            "numeric": plan.numeric,
            "categorical": plan.categorical,
            "datetime": plan.datetime,
            "excluded": plan.excluded,
        },
        "leaderboard": leaderboard,
        "best_model": {"key": best["key"], "name": best["name"]},
        "tuning": tuning,
        "evaluation": evaluation,
        "feature_importance": importances,
        "overfit_gap": safe_float(overfit_gap),
        "notes": notes,
        "input_schema": _input_schema(X_train, plan),
    }
    return TrainResult(pipeline=best_pipe, summary=summary, plan=plan, task=config.task, target=config.target)


def _tune(pipe, spec: ModelSpec, X, y, cv, scoring: str, before: float, config: TrainConfig) -> dict[str, Any]:
    """Randomised hyper-parameter search; ``before`` is the default model's CV score (higher is better)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            pipe,
            spec.search_space,
            n_iter=config.tune_iterations,
            cv=cv,
            scoring=scoring,
            random_state=config.random_state,
            error_score=np.nan,
        )
        search.fit(X, y)
    improved = bool(before is None or search.best_score_ > before)
    return {
        "iterations": config.tune_iterations,
        "score_before": safe_float(before),
        "score_after": safe_float(search.best_score_),
        "improved": improved,
        "best_params": {k.replace("model__", ""): _plain(v) for k, v in search.best_params_.items()},
        "estimator": clone(search.best_estimator_),
    }


def _plain(value):
    if isinstance(value, (np.floating, float)):
        return round(float(value), 5)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _evaluate_classifier(pipe, X_train, y_train, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    labels = [str(c) for c in pipe.classes_]
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "baseline_accuracy": accuracy_score(y_test, baseline.predict(X_test)),
        "train_accuracy": accuracy_score(y_train, pipe.predict(X_train)),
    }
    roc = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        try:
            if len(labels) == 2:
                positive = labels[1]
                metrics["roc_auc"] = roc_auc_score(y_test == positive, proba[:, 1])
                fpr, tpr, _ = roc_curve(y_test == positive, proba[:, 1])
                step = max(1, len(fpr) // 100)
                roc = {
                    "positive_label": positive,
                    "fpr": [round(float(v), 4) for v in fpr[::step]] + [1.0],
                    "tpr": [round(float(v), 4) for v in tpr[::step]] + [1.0],
                }
            else:
                metrics["roc_auc"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted", labels=labels)
        except ValueError:
            pass  # a class missing from the test split makes AUC undefined

    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    per_class = [
        {"label": label, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
    ]
    return {
        "metrics": {k: safe_float(v) for k, v in metrics.items()},
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist()},
        "per_class": per_class,
        "roc_curve": roc,
    }


def _evaluate_regressor(pipe, X_train, y_train, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    baseline = DummyRegressor().fit(X_train, y_train).predict(X_test)
    y_true = np.asarray(y_test, dtype=float)
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))) if nonzero.any() else None
    residuals = y_true - y_pred
    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_ae": median_absolute_error(y_true, y_pred),
        "mape": mape,
        "baseline_mae": mean_absolute_error(y_true, baseline),
        "baseline_r2": r2_score(y_true, baseline),
        "train_r2": r2_score(y_train, pipe.predict(X_train)),
    }
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_true), size=min(400, len(y_true)), replace=False)
    counts, edges = np.histogram(residuals, bins=20)
    return {
        "metrics": {k: safe_float(v) for k, v in metrics.items()},
        "scatter": [{"actual": float(y_true[i]), "predicted": float(y_pred[i])} for i in sorted(idx)],
        "residual_histogram": {"edges": edges.tolist(), "counts": counts.tolist()},
    }


def _permutation_importance(pipe, X_test, y_test, scoring: str, random_state: int) -> list[dict]:
    """Model-agnostic importance on the original (pre-encoding) columns."""
    if len(X_test) > 1500:
        X_test = X_test.sample(1500, random_state=random_state)
        y_test = y_test.loc[X_test.index]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            pipe, X_test, y_test, scoring=scoring, n_repeats=5, random_state=random_state, n_jobs=None
        )
    positive_total = float(np.clip(result.importances_mean, 0, None).sum()) or 1.0
    rows = [
        {
            "feature": col,
            "importance": float(mean),
            "std": float(std),
            "share": 100.0 * max(float(mean), 0.0) / positive_total,
        }
        for col, mean, std in zip(X_test.columns, result.importances_mean, result.importances_std, strict=True)
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def _input_schema(X: pd.DataFrame, plan: FeaturePlan) -> list[dict]:
    """Describe each feature so the UI can render a what-if prediction form."""
    schema = []
    for col in plan.numeric:
        values = X[col].dropna().astype(float)
        schema.append(
            {
                "name": col,
                "kind": "numeric",
                "min": safe_float(values.min()),
                "max": safe_float(values.max()),
                "default": safe_float(values.median()),
                "integer": bool((values % 1 == 0).all()),
            }
        )
    for col in plan.categorical:
        counts = X[col].dropna().astype(str).value_counts()
        schema.append(
            {
                "name": col,
                "kind": "categorical",
                "options": counts.head(40).index.tolist(),
                "default": counts.index[0] if len(counts) else None,
            }
        )
    for col in plan.datetime:
        values = pd.to_datetime(X[col], errors="coerce").dropna()
        schema.append(
            {
                "name": col,
                "kind": "datetime",
                "default": values.median().date().isoformat() if len(values) else None,
            }
        )
    return schema
