import io
import json

import pandas as pd
import pytest

from datalens.io import load_dataframe, prepare_dataframe
from datalens.utils import DataLensError


def test_csv_with_semicolon_delimiter_is_sniffed():
    data = b"a;b;c\n1;x;2020-01-01\n2;y;2020-02-01\n3;z;2020-03-01\n"
    df = load_dataframe(data, "file.csv")
    assert list(df.columns) == ["a", "b", "c"]
    assert pd.api.types.is_integer_dtype(df["a"])
    assert pd.api.types.is_datetime64_any_dtype(df["c"])


def test_numeric_strings_with_currency_and_separators_are_coerced():
    data = b'price,qty\n"$1,200.50",3\n"$99.00",1\n"(15.00)",2\n'
    df = load_dataframe(data, "prices.csv")
    assert df["price"].tolist() == [1200.5, 99.0, -15.0]


def test_null_tokens_become_missing_and_names_are_cleaned():
    data = b" name ,Unnamed: 1,name\nalice,N/A,1\n-,?,2\n"
    df = load_dataframe(data, "x.csv")
    assert list(df.columns) == ["name", "column_2", "name_2"]
    assert df["name"].isna().sum() == 1
    assert df["column_2"].isna().all()


def test_latin1_encoded_file_is_decoded():
    data = "city,value\nMünchen,1\nZürich,2\n".encode("cp1252")
    df = load_dataframe(data, "eu.csv")
    assert df["city"].tolist() == ["München", "Zürich"]


def test_json_records_and_wrapped_payloads():
    records = [{"a": 1, "b": {"c": 2}}, {"a": 3, "b": {"c": 4}}]
    df = load_dataframe(json.dumps(records).encode(), "r.json")
    assert list(df.columns) == ["a", "b.c"]
    wrapped = load_dataframe(json.dumps({"data": records}).encode(), "w.json")
    assert len(wrapped) == 2


def test_excel_round_trip():
    buffer = io.BytesIO()
    pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_excel(buffer, index=False)
    df = load_dataframe(buffer.getvalue(), "book.xlsx")
    assert df.shape == (2, 2)


@pytest.mark.parametrize(
    "data,name,message",
    [
        (b"", "empty.csv", "empty"),
        (b"a,b\n1,2\n", "file.exe", "Unsupported file type"),
        (b"a,b\n,\n,\n", "blank.csv", "no data rows"),
        (b"[1, 2, 3", "bad.json", "Could not parse"),
    ],
)
def test_invalid_files_raise_friendly_errors(data, name, message):
    with pytest.raises(DataLensError, match=message):
        load_dataframe(data, name)


def test_boolean_objects_become_boolean_dtype():
    df = prepare_dataframe(pd.DataFrame({"flag": [True, False, None]}, dtype=object))
    assert str(df["flag"].dtype) == "boolean"
