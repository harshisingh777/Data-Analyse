from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------- Helpers ----------
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
        if df[c].nunique() <= min(20, max(2, int(len(df) * 0.05))):
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


def fmt_pct(x: float) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "N/A"


def clean_feat_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")


def to_percentage_importances(feature_importances):
    cleaned = []
    for feat, imp in feature_importances:
        cleaned.append((clean_feat_name(feat), float(imp) * 100.0))
    return cleaned


def generate_analysis_text(task, target, raw_metrics, feature_importances_pct):
    text = f"Target Variable: {target}\n"

    if task == "classification":
        acc = raw_metrics.get("accuracy", 0.0)
        base_acc = raw_metrics.get("baseline_accuracy", 0.0)
        cv_mean = raw_metrics.get("cv_mean", 0.0)
        f1 = raw_metrics.get("f1_weighted", 0.0)

        text += (
            f"The model predicts **{target}** with **{fmt_pct(acc)} accuracy**, "
            f"well above a simple baseline at {fmt_pct(base_acc)}. "
            f"In repeated testing (cross-validation), it stayed around **{fmt_pct(cv_mean)}**, "
            f"indicating reliable performance.\n\n"
            f"The overall balance of precision and recall (F1-score) is **{fmt_pct(f1)}**.\n"
        )

    else:
        r2 = raw_metrics.get("r2", 0.0)
        base_r2 = raw_metrics.get("baseline_r2", 0.0)
        cv_mean = raw_metrics.get("cv_mean", 0.0)

        text += (
            f"The model achieves an **R² of {fmt_pct(r2)}**, "
            f"meaning it explains roughly {fmt_pct(r2)} of the variation in **{target}**. "
            f"This compares to a baseline R² of {fmt_pct(base_r2)}. "
            f"Cross-validation averages **{fmt_pct(cv_mean)}**, suggesting stable performance.\n"
        )

    if feature_importances_pct:
        text += "\n**Key factors influencing predictions:**\n"
        for feat, imp_pct in feature_importances_pct:
            text += f"- {feat}: contributes ~{imp_pct:.2f}% to the prediction\n"

        most_feat, most_imp = max(feature_importances_pct, key=lambda x: x[1])
        least_feat, least_imp = min(feature_importances_pct, key=lambda x: x[1])
        text += (
            f"\n📌 **Fact:** {most_feat} is the most important factor (~{most_imp:.2f}%), "
            f"while {least_feat} has the least influence (~{least_imp:.2f}%) in this dataset."
        )

    return text


# ---------- API ----------
@app.route("/autotrain", methods=["POST"])
def auto_train():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith((".xls", ".xlsx")):
            df = pd.read_excel(filepath)
        elif filepath.endswith(".json"):
            df = pd.read_json(filepath)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        target = infer_target(df)
        if not target:
            return jsonify({"error": "Could not infer target column"}), 400

        task = task_type(df[target])
        X_full = df.drop(columns=[target])
        y = df[target]

        pre, _ = build_preprocessor(df, target)

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

        pipe_dummy.fit(X_train, y_train)
        pipe_main.fit(X_train, y_train)

        raw = {}
        if task == "classification":
            y_pred = pipe_main.predict(X_test)
            raw["baseline_accuracy"] = accuracy_score(y_test, pipe_dummy.predict(X_test))
            raw["accuracy"] = accuracy_score(y_test, y_pred)
            raw["f1_weighted"] = f1_score(y_test, y_pred, average="weighted")
            cv_scores = cross_val_score(pipe_main, X_full, y, cv=5, scoring=scoring)
            raw["cv_mean"], raw["cv_std"] = float(np.mean(cv_scores)), float(np.std(cv_scores))
        else:
            y_pred = pipe_main.predict(X_test)
            raw["baseline_r2"] = r2_score(y_test, pipe_dummy.predict(X_test))
            raw["r2"] = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(pipe_main, X_full, y, cv=5, scoring=scoring)
            raw["cv_mean"], raw["cv_std"] = float(np.mean(cv_scores)), float(np.std(cv_scores))

        feature_importances_pct = []
        try:
            pre.fit(X_full)
            feat_names = pre.get_feature_names_out()
            model = pipe_main.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                pairs = list(zip(feat_names, model.feature_importances_))
                pairs.sort(key=lambda x: x[1], reverse=True)
                feature_importances_pct = to_percentage_importances(pairs[:20])
        except Exception:
            pass

        analysis_text = generate_analysis_text(task, target, raw, feature_importances_pct)

        # Always create fun fact separately
        if task == "classification":
            acc = raw.get("accuracy", 0.0)
            base_acc = raw.get("baseline_accuracy", 0.0)
            times = (acc / base_acc) if base_acc > 0 else 0
            fun_fact = f"Your model is about {times:.1f}× more accurate than a baseline guesser."
        else:
            r2 = raw.get("r2", 0.0)
            fun_fact = f"Your model explains about {fmt_pct(r2)} of the dataset’s variability."

        print("\n--- FULL ANALYSIS TEXT ---\n", analysis_text)
        print("\n--- FUN FACT ---\n", fun_fact)

        if task == "classification":
            metrics = {
                "accuracy": fmt_pct(raw["accuracy"]),
                "baseline_accuracy": fmt_pct(raw["baseline_accuracy"]),
                "cv_accuracy_mean": fmt_pct(raw["cv_mean"]),
                "cv_accuracy_std": fmt_pct(raw["cv_std"]),
                "f1_weighted": fmt_pct(raw["f1_weighted"]),
            }
        else:
            metrics = {
                "r2": fmt_pct(raw["r2"]),
                "baseline_r2": fmt_pct(raw["baseline_r2"]),
                "cv_r2_mean": fmt_pct(raw["cv_mean"]),
                "cv_r2_std": fmt_pct(raw["cv_std"]),
            }

        return jsonify({
            "message": "Training complete",
            "target": target,
            "task": task,
            "metrics": metrics,
            "feature_importances": feature_importances_pct,
            "analysis_text": analysis_text,
            "fun_fact": fun_fact
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)