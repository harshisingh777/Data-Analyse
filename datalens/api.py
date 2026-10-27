"""Flask application: JSON API plus the single-page front end."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from . import __version__
from .config import Settings
from .insights import dataset_insights, executive_summary, model_insights, statistics_insights
from .io import SUPPORTED_EXTENSIONS, load_dataframe
from .modeling import TrainConfig, model_catalogue, train_models
from .profiling import NUMERIC
from .report import render_report
from .samples import list_samples, load_sample
from .sql import TABLE_NAME, example_queries
from .stats import association_tests, normality_tests
from .store import DatasetStore
from .target import class_balance, infer_task, validate_target
from .utils import DataLensError, to_jsonable

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
log = logging.getLogger("datalens")


def _json(payload, status: int = 200) -> Response:
    response = jsonify(to_jsonable(payload))
    response.status_code = status
    return response


def _safe_filename(name: str, suffix: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_") or "dataset"
    return f"{stem}{suffix}"


def create_app(store: DatasetStore | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = Settings.MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    datasets = store or DatasetStore()
    app.extensions["datalens_store"] = datasets

    # ------------------------------------------------------------------ errors
    @app.errorhandler(DataLensError)
    def handle_datalens_error(exc: DataLensError):
        return _json({"error": str(exc)}, exc.status_code)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_exc):
        return _json({"error": f"File is larger than the {Settings.MAX_UPLOAD_MB} MB limit."}, 413)

    @app.errorhandler(HTTPException)
    def handle_http(exc: HTTPException):
        if request.path.startswith("/api/"):
            return _json({"error": exc.description}, exc.code or 500)
        return exc

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        log.exception("Unhandled error")
        return _json({"error": f"Unexpected server error: {exc}"}, 500)

    @app.after_request
    def security_headers(response: Response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    # ------------------------------------------------------------------ front end
    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(STATIC_DIR, filename)

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return _json({"status": "ok", "version": __version__, "datasets_loaded": len(datasets)})

    @app.get("/api/config")
    def config():
        return _json(
            {
                "version": __version__,
                "max_upload_mb": Settings.MAX_UPLOAD_MB,
                "supported_extensions": SUPPORTED_EXTENSIONS,
                "models": {
                    task: [{"key": k, "name": s.name} for k, s in model_catalogue(task).items()]
                    for task in ("classification", "regression")
                },
                # Task-agnostic choices for the UI; "linear" resolves to logistic or ridge regression.
                "model_choices": [
                    {"key": "linear", "name": "Linear (logistic / ridge)"},
                    {"key": "decision_tree", "name": "Decision tree"},
                    {"key": "random_forest", "name": "Random forest"},
                    {"key": "gradient_boosting", "name": "Gradient boosting"},
                    {"key": "knn", "name": "k-nearest neighbours"},
                ],
            }
        )

    @app.get("/api/samples")
    def samples():
        return _json({"samples": list_samples()})

    # ------------------------------------------------------------------ datasets
    def _dataset_payload(session) -> dict:
        return {
            "dataset": session.meta(),
            "profile": session.profile,
            "insights": dataset_insights(session.profile),
            "summary": executive_summary(session.profile, session.model.summary if session.model else None),
        }

    @app.post("/api/datasets")
    def upload_dataset():
        file = request.files.get("file")
        if file is None or not file.filename:
            raise DataLensError("No file was uploaded. Attach a file in the 'file' field.")
        start = time.perf_counter()
        df = load_dataframe(file.read(), file.filename)
        session = datasets.add(file.filename, df, source="upload")
        log.info("Loaded %s (%d×%d) in %.2fs", file.filename, *df.shape, time.perf_counter() - start)
        return _json(_dataset_payload(session), 201)

    @app.post("/api/samples/<key>")
    def load_sample_dataset(key: str):
        name, df = load_sample(key)
        session = datasets.add(name, df, source="sample")
        return _json(_dataset_payload(session), 201)

    @app.get("/api/datasets/<dataset_id>")
    def get_dataset(dataset_id: str):
        return _json(_dataset_payload(datasets.get(dataset_id)))

    @app.delete("/api/datasets/<dataset_id>")
    def delete_dataset(dataset_id: str):
        datasets.remove(dataset_id)
        return _json({"deleted": dataset_id})

    @app.get("/api/datasets/<dataset_id>/rows")
    def preview_rows(dataset_id: str):
        session = datasets.get(dataset_id)
        offset = max(request.args.get("offset", 0, type=int), 0)
        limit = min(max(request.args.get("limit", 50, type=int), 1), 500)
        sort = request.args.get("sort")
        df = session.df
        if sort and sort in df.columns:
            df = df.sort_values(sort, ascending=request.args.get("order", "asc") != "desc", na_position="last")
        page = df.iloc[offset : offset + limit]
        rows = page.astype(object).where(page.notna(), None).to_numpy().tolist()
        return _json({"columns": list(df.columns), "rows": rows, "offset": offset, "total": int(len(df))})

    @app.get("/api/datasets/<dataset_id>/statistics")
    def statistics(dataset_id: str):
        session = datasets.get(dataset_id)
        suggested = session.suggested_target() or {}
        target = request.args.get("target") or suggested.get("column")
        if not target:
            raise DataLensError("Choose a target column for the statistical tests.")
        if target not in session.df.columns:
            raise DataLensError(f"Column '{target}' does not exist.")
        task = request.args.get("task") or infer_task(session.df[target])
        validate_target(session.df, target, task)
        alpha = request.args.get("alpha", 0.05, type=float)
        if not 0 < alpha < 0.5:
            raise DataLensError("alpha must be between 0 and 0.5.")
        result = association_tests(session.df, target, task, session.semantic, alpha=alpha)
        numeric_cols = [c for c, t in session.semantic.items() if t == NUMERIC]
        result["normality"] = normality_tests(session.df, numeric_cols, alpha=alpha)
        result["class_balance"] = class_balance(session.df[target]) if task == "classification" else None
        result["insights"] = statistics_insights(result)
        session.statistics = result
        return _json(result)

    @app.post("/api/datasets/<dataset_id>/train")
    def train(dataset_id: str):
        session = datasets.get(dataset_id)
        body = request.get_json(silent=True) or {}
        suggested = session.suggested_target() or {}
        target = body.get("target") or suggested.get("column")
        if not target:
            raise DataLensError("Choose a target column to train a model.")
        try:
            config = TrainConfig(
                target=target,
                task=body.get("task") or None,
                models=body.get("models") or None,
                test_size=float(body.get("test_size", 0.2)),
                cv_folds=int(body.get("cv_folds", 5)),
                tune=bool(body.get("tune", False)),
            )
        except (TypeError, ValueError) as exc:
            raise DataLensError(f"Invalid training options: {exc}") from exc
        start = time.perf_counter()
        result = train_models(session.df, config, session.semantic)
        result.summary["duration_seconds"] = round(time.perf_counter() - start, 2)
        result.summary["insights"] = model_insights(result.summary)
        session.model = result
        log.info(
            "Trained %s on %s in %.2fs",
            result.summary["best_model"]["name"],
            session.name,
            result.summary["duration_seconds"],
        )
        return _json(result.summary)

    @app.get("/api/datasets/<dataset_id>/model")
    def model_summary(dataset_id: str):
        session = datasets.get(dataset_id)
        if session.model is None:
            raise DataLensError("No model has been trained for this dataset yet.", 404)
        return _json(session.model.summary)

    @app.post("/api/datasets/<dataset_id>/predict")
    def predict(dataset_id: str):
        session = datasets.get(dataset_id)
        if session.model is None:
            raise DataLensError("Train a model before making predictions.")
        body = request.get_json(silent=True) or {}
        records = body.get("records") or ([body["record"]] if isinstance(body.get("record"), dict) else None)
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise DataLensError("Send {'record': {...}} or {'records': [{...}, ...]}.")
        if len(records) > 1000:
            raise DataLensError("At most 1,000 records can be scored per request.")
        return _json({"predictions": session.model.predict(records)})

    @app.get("/api/datasets/<dataset_id>/model/download")
    def download_model(dataset_id: str):
        session = datasets.get(dataset_id)
        if session.model is None:
            raise DataLensError("No model has been trained for this dataset yet.", 404)
        filename = _safe_filename(session.name, "_model.joblib")
        return Response(
            session.model.to_bytes(),
            mimetype="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/datasets/<dataset_id>/sql/schema")
    def sql_schema(dataset_id: str):
        session = datasets.get(dataset_id)
        return _json(
            {
                "table": TABLE_NAME,
                "columns": session.sql.schema(),
                "examples": example_queries(session.profile),
            }
        )

    @app.post("/api/datasets/<dataset_id>/sql")
    def run_sql(dataset_id: str):
        session = datasets.get(dataset_id)
        body = request.get_json(silent=True) or {}
        return _json(session.sql.execute(str(body.get("query", ""))))

    @app.get("/api/datasets/<dataset_id>/report")
    def report(dataset_id: str):
        session = datasets.get(dataset_id)
        html = render_report(session)
        headers = {}
        if request.args.get("download"):
            filename = _safe_filename(session.name, "_report.html")
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return Response(html, mimetype="text/html", headers=headers)

    @app.get("/api/datasets/<dataset_id>/export")
    def export_csv(dataset_id: str):
        session = datasets.get(dataset_id)
        filename = _safe_filename(session.name, "_clean.csv")
        csv = session.df.replace({np.nan: None}).to_csv(index=False)
        return Response(csv, mimetype="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return app
