# train_models.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -----------------------
# Helpers
# -----------------------
def infer_target(df: pd.DataFrame):
    priority = ["target", "label", "y", "class", "outcome", "Crop"]
    for p in priority:
        for c in df.columns:
            if c.lower() == p.lower():
                return c
    last = df.columns[-1]
    if not ("id" in last.lower() or df[last].nunique() == len(df)):
        return last
    for c in df.columns:
        if df[c].nunique() <= min(20, max(2, int(len(df)*0.05))):
            return c
    return None

def task_type(series: pd.Series):
    if series.dtype == "O" or series.nunique() <= 20:
        return "classification"
    return "regression"

def build_preprocessor(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, X

# -----------------------
# Main training function
# -----------------------
def train_and_analyze(file_path):
    print(f"Loading dataset from: {file_path}")
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    target = infer_target(df)
    if not target:
        raise ValueError("Could not infer a target column.")

    print(f"Target column: {target}")
    task = task_type(df[target])
    print(f"Detected task: {task}")

    X_full = df.drop(columns=[target])
    y = df[target]

    pre, X = build_preprocessor(df, target)

    if task == "classification":
        main_est = RandomForestClassifier(n_estimators=200, random_state=42)
        dummy = DummyClassifier(strategy="most_frequent")
        scoring = "accuracy"
    else:
        main_est = RandomForestRegressor(n_estimators=300, random_state=42)
        dummy = DummyRegressor(strategy="mean")
        scoring = "r2"

    pipe_main = Pipeline([("pre", pre), ("model", main_est)])
    pipe_dummy = Pipeline([("pre", pre), ("model", dummy)])

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42,
        stratify=y if task == "classification" else None
    )

    # Fit models
    pipe_dummy.fit(X_train, y_train)
    pipe_main.fit(X_train, y_train)

    # Evaluate
    if task == "classification":
        preds = pipe_main.predict(X_test)
        metrics = {
            "baseline_accuracy": accuracy_score(y_test, pipe_dummy.predict(X_test)),
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
            "cv_accuracy": {
                "mean": np.mean(cross_val_score(pipe_main, X_full, y, cv=5, scoring=scoring)),
                "std": np.std(cross_val_score(pipe_main, X_full, y, cv=5, scoring=scoring))
            }
        }
    else:
        preds = pipe_main.predict(X_test)
        metrics = {
            "baseline_r2": r2_score(y_test, pipe_dummy.predict(X_test)),
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "cv_r2": {
                "mean": np.mean(cross_val_score(pipe_main, X_full, y, cv=5, scoring=scoring)),
                "std": np.std(cross_val_score(pipe_main, X_full, y, cv=5, scoring=scoring))
            }
        }

    print("Metrics:", metrics)

    # Feature importance
    try:
        pre.fit(X_full)
        feat_names = pre.get_feature_names_out()
        model = pipe_main.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = sorted(
                zip(feat_names, model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:20]
            print("\nTop Feature Importances:")
            for feat, imp in importances:
                print(f"{feat}: {imp:.4f}")
    except Exception as e:
        print("Could not extract feature importances:", e)

if __name__ == "__main__":
    path = input("Enter dataset file path: ").strip()
    train_and_analyze(path)