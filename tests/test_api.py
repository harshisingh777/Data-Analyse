import io

import joblib
import pytest

from datalens.store import DatasetStore
from datalens.utils import DataLensError

from .conftest import csv_bytes


def upload(client, df, name="data.csv"):
    return client.post(
        "/api/datasets", data={"file": (io.BytesIO(csv_bytes(df)), name)}, content_type="multipart/form-data"
    )


@pytest.fixture
def dataset_id(client, classification_df):
    response = upload(client, classification_df, "churn.csv")
    assert response.status_code == 201
    return response.get_json()["dataset"]["id"]


def test_index_and_static_assets_are_served(client):
    page = client.get("/").data
    assert b"DataLens" in page
    assert b"Made by <strong>Harshi Singh</strong>" in page
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/static/css/app.css").status_code == 200


def test_health_and_config(client):
    assert client.get("/api/health").get_json()["status"] == "ok"
    config = client.get("/api/config").get_json()
    assert ".csv" in config["supported_extensions"]
    assert {m["key"] for m in config["model_choices"]} >= {"linear", "random_forest"}


def test_upload_returns_profile_and_suggested_target(client, classification_df):
    body = upload(client, classification_df).get_json()
    assert body["dataset"]["rows"] == 300
    assert body["dataset"]["suggested_target"] == {"column": "churn", "task": "classification"}
    assert body["profile"]["overview"]["columns"] == 7
    assert body["insights"] and body["summary"].startswith("The dataset contains 300 records")


def test_upload_errors(client):
    assert client.post("/api/datasets").status_code == 400
    bad = client.post(
        "/api/datasets", data={"file": (io.BytesIO(b"x"), "evil.exe")}, content_type="multipart/form-data"
    )
    assert bad.status_code == 400
    assert "Unsupported" in bad.get_json()["error"]


def test_upload_size_limit(app, client):
    app.config["MAX_CONTENT_LENGTH"] = 100
    response = client.post(
        "/api/datasets",
        data={"file": (io.BytesIO(b"a,b\n" + b"1,2\n" * 100), "big.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert "limit" in response.get_json()["error"]


def test_unknown_dataset_is_404(client):
    response = client.get("/api/datasets/doesnotexist")
    assert response.status_code == 404
    assert "error" in response.get_json()


def test_rows_paging_and_sorting(client, dataset_id):
    body = client.get(f"/api/datasets/{dataset_id}/rows?offset=10&limit=5&sort=tickets&order=desc").get_json()
    assert body["total"] == 300 and body["offset"] == 10 and len(body["rows"]) == 5
    tickets = [r[body["columns"].index("tickets")] for r in body["rows"]]
    assert tickets == sorted(tickets, reverse=True)


def test_statistics_endpoint(client, dataset_id):
    body = client.get(f"/api/datasets/{dataset_id}/statistics?target=churn&alpha=0.05").get_json()
    assert body["target"] == "churn" and body["significant_count"] >= 1
    assert body["class_balance"][0]["count"] > 0
    assert body["normality"]
    assert client.get(f"/api/datasets/{dataset_id}/statistics?target=nope").status_code == 400
    assert client.get(f"/api/datasets/{dataset_id}/statistics?target=churn&alpha=2").status_code == 400


def test_train_predict_and_download(client, dataset_id):
    assert client.post(f"/api/datasets/{dataset_id}/predict", json={"record": {}}).status_code == 400

    response = client.post(
        f"/api/datasets/{dataset_id}/train",
        json={"target": "churn", "models": ["linear", "decision_tree"], "cv_folds": 3},
    )
    assert response.status_code == 200
    summary = response.get_json()
    assert summary["best_model"]["key"] in ("logistic_regression", "decision_tree")
    assert summary["insights"] and summary["duration_seconds"] >= 0

    assert client.get(f"/api/datasets/{dataset_id}/model").get_json()["target"] == "churn"
    assert client.get(f"/api/datasets/{dataset_id}").get_json()["dataset"]["has_model"] is True

    pred = client.post(
        f"/api/datasets/{dataset_id}/predict", json={"record": {"tenure": 5, "tickets": 4, "contract": "monthly"}}
    ).get_json()
    assert pred["predictions"][0]["prediction"] in ("Yes", "No")
    assert client.post(f"/api/datasets/{dataset_id}/predict", json={"records": "bad"}).status_code == 400

    model = client.get(f"/api/datasets/{dataset_id}/model/download")
    assert "attachment" in model.headers["Content-Disposition"]
    assert joblib.load(io.BytesIO(model.data))["task"] == "classification"


def test_train_rejects_invalid_options(client, dataset_id):
    assert (
        client.post(f"/api/datasets/{dataset_id}/train", json={"target": "churn", "cv_folds": "x"}).status_code == 400
    )
    assert (
        client.post(f"/api/datasets/{dataset_id}/train", json={"target": "churn", "models": ["svm"]}).status_code == 400
    )


def test_sql_endpoints(client, dataset_id):
    schema = client.get(f"/api/datasets/{dataset_id}/sql/schema").get_json()
    assert schema["table"] == "dataset" and schema["examples"]
    ok = client.post(f"/api/datasets/{dataset_id}/sql", json={"query": "SELECT COUNT(*) AS n FROM dataset"})
    assert ok.get_json()["rows"] == [[300]]
    blocked = client.post(f"/api/datasets/{dataset_id}/sql", json={"query": "DROP TABLE dataset"})
    assert blocked.status_code == 400


def test_report_and_export(client, dataset_id):
    client.get(f"/api/datasets/{dataset_id}/statistics?target=churn")
    client.post(f"/api/datasets/{dataset_id}/train", json={"target": "churn", "models": ["linear"], "cv_folds": 3})
    report = client.get(f"/api/datasets/{dataset_id}/report?download=1")
    html = report.get_data(as_text=True)
    assert report.mimetype == "text/html" and "attachment" in report.headers["Content-Disposition"]
    for section in (
        "Executive summary",
        "Data quality",
        "Statistical tests against churn",
        "Model leaderboard",
        "Confusion matrix",
    ):
        assert section in html
    export = client.get(f"/api/datasets/{dataset_id}/export")
    assert export.mimetype == "text/csv"
    assert export.get_data(as_text=True).splitlines()[0].startswith("customer_id,signup")


def test_report_escapes_html_in_data(client):
    import pandas as pd

    df = pd.DataFrame({"<script>alert(1)</script>": ["a", "b"] * 15, "y": range(30)})
    dataset_id = upload(client, df).get_json()["dataset"]["id"]
    html = client.get(f"/api/datasets/{dataset_id}/report").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_samples_list_and_load(client):
    samples = client.get("/api/samples").get_json()["samples"]
    assert {s["key"] for s in samples} == {"customer_churn", "house_prices", "retail_orders"}
    body = client.post("/api/samples/house_prices").get_json()
    assert body["dataset"]["suggested_target"]["column"] == "price"
    assert client.post("/api/samples/unknown").status_code == 404


def test_store_evicts_least_recently_used(classification_df):
    store = DatasetStore(capacity=2)
    a = store.add("a", classification_df)
    b = store.add("b", classification_df)
    store.get(a.id)  # a is now most recently used
    store.add("c", classification_df)
    assert store.get(a.id)
    with pytest.raises(DataLensError, match="not found"):
        store.get(b.id)


def test_delete_dataset(client, dataset_id):
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 200
    assert client.get(f"/api/datasets/{dataset_id}").status_code == 404


def test_security_headers(client):
    response = client.get("/api/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_upload_ids_are_content_addressed_and_keep_models(client, classification_df):
    first = upload(client, classification_df, "churn.csv").get_json()["dataset"]["id"]
    client.post(f"/api/datasets/{first}/train", json={"target": "churn", "models": ["linear"], "cv_folds": 3})
    again = upload(client, classification_df, "churn.csv").get_json()["dataset"]
    assert again["id"] == first
    assert again["has_model"] is True
    other = upload(client, classification_df.head(100), "churn.csv").get_json()["dataset"]["id"]
    assert other != first


def test_sample_datasets_rebuild_on_a_fresh_instance(classification_df):
    """Simulates a serverless request reaching an instance that never loaded the sample."""
    from datalens.api import create_app

    loaded = create_app(DatasetStore(capacity=2)).test_client().post("/api/samples/customer_churn").get_json()
    dataset_id = loaded["dataset"]["id"]
    assert dataset_id == "sample-customer_churn"

    fresh = create_app(DatasetStore(capacity=2)).test_client()
    body = fresh.get(f"/api/datasets/{dataset_id}").get_json()
    assert body["dataset"]["rows"] == loaded["dataset"]["rows"]
    assert fresh.get("/api/datasets/sample-unknown").status_code == 404
    assert fresh.get("/api/datasets/abc123").status_code == 404


def test_vercel_entry_point_exposes_wsgi_app():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("vercel_index", Path(__file__).parent.parent / "api" / "index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.app.test_client().get("/api/health").status_code == 200
