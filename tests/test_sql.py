import pandas as pd
import pytest

from datalens.profiling import profile_dataframe
from datalens.sql import SqlWorkbench, example_queries
from datalens.utils import DataLensError


@pytest.fixture
def workbench(classification_df):
    wb = SqlWorkbench(classification_df, row_limit=50, timeout_seconds=1)
    yield wb
    wb.close()


def test_select_with_aggregates_and_window_functions(workbench, classification_df):
    result = workbench.execute(
        "SELECT contract, COUNT(*) AS n, RANK() OVER (ORDER BY COUNT(*) DESC) AS r "
        "FROM dataset GROUP BY contract ORDER BY n DESC;"
    )
    assert result["columns"] == ["contract", "n", "r"]
    expected = classification_df["contract"].value_counts()
    assert {row[0]: row[1] for row in result["rows"]} == expected.to_dict()
    assert result["rows"][0][2] == 1


def test_dates_are_stored_as_iso_text(workbench):
    result = workbench.execute("SELECT MIN(signup), strftime('%Y', MAX(signup)) FROM dataset")
    assert result["rows"][0][0] == "2022-01-01 00:00:00"


def test_row_limit_truncates(workbench):
    result = workbench.execute("SELECT * FROM dataset")
    assert result["row_count"] == 50
    assert result["truncated"] is True


@pytest.mark.parametrize(
    "query",
    [
        "DROP TABLE dataset",
        "DELETE FROM dataset",
        "INSERT INTO dataset (tenure) VALUES (1)",
        "UPDATE dataset SET tenure = 0",
        "PRAGMA table_info(dataset)",
        "ATTACH DATABASE 'evil.db' AS evil",
        "CREATE TABLE t (x INT)",
        "SELECT 1; DROP TABLE dataset",
    ],
)
def test_write_and_admin_statements_are_blocked(workbench, query):
    with pytest.raises(DataLensError):
        workbench.execute(query)
    assert workbench.execute("SELECT COUNT(*) FROM dataset")["rows"][0][0] == 300


def test_runaway_queries_time_out(workbench):
    with pytest.raises(DataLensError, match="time limit"):
        workbench.execute("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c")


def test_empty_and_invalid_queries(workbench):
    with pytest.raises(DataLensError, match="Enter a SQL query"):
        workbench.execute("   ")
    with pytest.raises(DataLensError, match="SQL error"):
        workbench.execute("SELECT nope FROM dataset")


def test_schema_and_examples_run(classification_df, workbench):
    names = [c["name"] for c in workbench.schema()]
    assert names == list(classification_df.columns)
    for example in example_queries(profile_dataframe(classification_df)):
        assert workbench.execute(example["sql"])["columns"]


def test_object_columns_with_mixed_values_load():
    df = pd.DataFrame({"mixed": [1, "a", None, 2.5], "flag": pd.array([True, False, None, True], dtype="boolean")})
    wb = SqlWorkbench(df)
    assert wb.execute("SELECT SUM(flag) FROM dataset")["rows"][0][0] == 2
