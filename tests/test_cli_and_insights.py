import json
import math

import numpy as np
import pandas as pd

from datalens.cli import main
from datalens.insights import dataset_insights, executive_summary
from datalens.profiling import profile_dataframe
from datalens.samples import SAMPLES, SAMPLES_DIR
from datalens.utils import pct, safe_float, to_jsonable


def test_to_jsonable_handles_numpy_pandas_and_nan():
    payload = {
        "i": np.int64(3),
        "f": np.float32(1.5),
        "nan": float("nan"),
        "inf": np.inf,
        "b": np.bool_(True),
        "ts": pd.Timestamp("2024-01-02"),
        "nat": pd.NaT,
        "arr": np.array([1, 2]),
        "nested": [{"x": np.nan}],
    }
    out = to_jsonable(payload)
    json.dumps(out, allow_nan=False)  # must be strict JSON
    assert out == {
        "i": 3,
        "f": 1.5,
        "nan": None,
        "inf": None,
        "b": True,
        "ts": "2024-01-02T00:00:00",
        "nat": None,
        "arr": [1, 2],
        "nested": [{"x": None}],
    }


def test_small_helpers():
    assert pct(1, 4) == 25.0 and pct(1, 0) == 0.0
    assert safe_float("2.555", 2) == 2.56 and safe_float(math.inf) is None and safe_float("x") is None


def test_insights_and_summary(classification_df):
    profile = profile_dataframe(classification_df)
    findings = dataset_insights(profile)
    assert findings[0]["kind"] == "quality"
    summary = executive_summary(profile)
    assert "300 records" in summary and "grade" in summary


def test_sample_files_exist_and_match_metadata():
    for meta in SAMPLES.values():
        df = pd.read_csv(SAMPLES_DIR / meta["file"])
        assert meta["target"] in df.columns
        assert len(df) > 1000


def test_cli_profile_and_train(tmp_path, capsys, regression_df):
    data = tmp_path / "houses.csv"
    regression_df.to_csv(data, index=False)
    out_json = tmp_path / "profile.json"
    assert main(["profile", str(data), "--json", str(out_json)]) == 0
    assert json.loads(out_json.read_text())["overview"]["rows"] == 250

    report, model = tmp_path / "report.html", tmp_path / "model.joblib"
    assert main(["train", str(data), "--cv", "3", "--report", str(report), "--model", str(model)]) == 0
    printed = capsys.readouterr().out
    assert "Training models to predict 'price'" in printed and "Hold-out metrics" in printed
    assert "Model leaderboard" in report.read_text(encoding="utf-8")
    assert model.stat().st_size > 0


def test_cli_reports_errors(tmp_path, capsys):
    assert main(["profile", str(tmp_path / "missing.csv")]) == 1
    assert "File not found" in capsys.readouterr().err
