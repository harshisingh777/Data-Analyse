import numpy as np
import pandas as pd
import pytest

from datalens.api import create_app
from datalens.store import DatasetStore


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """Small churn-like table with a learnable signal, missing values and mixed types."""
    rng = np.random.default_rng(0)
    n = 300
    tenure = rng.integers(1, 60, n)
    tickets = rng.poisson(1.5, n)
    contract = rng.choice(["monthly", "annual"], n)
    logit = -1 + 0.8 * tickets - 0.05 * tenure + 1.2 * (contract == "monthly")
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup": pd.date_range("2022-01-01", periods=n, freq="D"),
            "tenure": tenure,
            "tickets": tickets,
            "contract": contract,
            "noise": rng.normal(size=n),
            "churn": churn,
        }
    )
    df.loc[::17, "tenure"] = np.nan
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 250
    size = rng.normal(100, 25, n)
    rooms = rng.integers(1, 6, n)
    city = rng.choice(["A", "B", "C"], n)
    price = 3 * size + 20 * rooms + pd.Series(city).map({"A": 0, "B": 50, "C": 120}).to_numpy() + rng.normal(0, 10, n)
    return pd.DataFrame({"size": size, "rooms": rooms, "city": city, "price": price})


@pytest.fixture
def app():
    application = create_app(DatasetStore(capacity=4))
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
