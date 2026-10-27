"""Generate the deterministic demo datasets in data/samples/.

The data is synthetic but built from plausible business relationships, plus deliberate
imperfections (missing values, duplicates, outliers, collinear columns) so that every
part of the profiling and modelling pipeline has something real to find.

    python scripts/generate_samples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "samples"


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _with_missing(rng, values: pd.Series, rate: float) -> pd.Series:
    mask = rng.random(len(values)) < rate
    return values.mask(mask)


def customer_churn(rng: np.random.Generator, n: int = 2400) -> pd.DataFrame:
    plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.45, 0.35, 0.20])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["Credit card", "Bank transfer", "Electronic check", "PayPal"], n, p=[0.3, 0.25, 0.3, 0.15])
    region = rng.choice(["North", "South", "East", "West", "Central"], n, p=[0.24, 0.2, 0.2, 0.22, 0.14])
    tenure = np.clip(rng.gamma(2.0, 14, n).round(), 1, 72).astype(int)
    base = pd.Series(plan).map({"Basic": 29, "Standard": 55, "Premium": 89}).to_numpy()
    monthly = np.round(base + rng.normal(0, 6, n), 2)
    total = np.round(monthly * tenure * rng.normal(1, 0.03, n), 2)
    tickets = rng.poisson(1.2 + (plan == "Basic") * 0.6, n)
    session = np.round(rng.lognormal(3.2, 0.6, n), 1)
    session[rng.choice(n, 25, replace=False)] *= 8  # heavy users / logging glitches
    auto_renew = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    satisfaction = np.clip(np.round(4.2 - 0.35 * tickets + rng.normal(0, 0.8, n)), 1, 5)
    signup = pd.Timestamp("2025-01-01") - pd.to_timedelta(tenure * 30 + rng.integers(0, 30, n), unit="D")

    logit = (
        -2.2
        + 2.2 * (contract == "Month-to-month")
        - 1.4 * (contract == "Two year")
        + 0.7 * tickets
        - 0.06 * tenure
        - 0.9 * (satisfaction - 3)
        + 1.0 * (payment == "Electronic check")
        - 0.8 * (auto_renew == "Yes")
        + 0.02 * (monthly - 50)
    )
    churned = np.where(rng.random(n) < _sigmoid(logit), "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "signup_date": signup.strftime("%Y-%m-%d"),
            "region": region,
            "plan": plan,
            "contract": contract,
            "payment_method": payment,
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_tickets": tickets,
            "avg_session_minutes": session,
            "auto_renew": auto_renew,
            "satisfaction_score": satisfaction,
            "churned": churned,
        }
    )
    df["monthly_charges"] = _with_missing(rng, df["monthly_charges"], 0.03)
    df["satisfaction_score"] = _with_missing(rng, df["satisfaction_score"], 0.08)
    df["payment_method"] = _with_missing(rng, df["payment_method"], 0.02)
    duplicates = df.sample(12, random_state=1)
    return pd.concat([df, duplicates], ignore_index=True)


def house_prices(rng: np.random.Generator, n: int = 1800) -> pd.DataFrame:
    cities = {"Austin": 1.15, "Denver": 1.05, "Phoenix": 0.85, "Raleigh": 0.9, "Seattle": 1.45}
    city = rng.choice(list(cities), n, p=[0.22, 0.2, 0.22, 0.18, 0.18])
    area = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.35, 0.5, 0.15])
    sqft = np.clip(rng.normal(1850, 620, n), 550, 5200).round()
    bedrooms = np.clip(np.round(sqft / 650 + rng.normal(0, 0.6, n)), 1, 7).astype(int)
    bathrooms = np.clip(np.round((bedrooms * 0.7 + rng.normal(0, 0.4, n)) * 2) / 2, 1, 5)
    year_built = rng.integers(1950, 2024, n)
    lot = np.round(rng.lognormal(-1.6, 0.6, n) * np.where(area == "Rural", 4, 1), 3)
    garage = rng.choice([0, 1, 2, 3], n, p=[0.12, 0.33, 0.45, 0.10])
    condition = rng.choice(["Poor", "Fair", "Good", "Excellent"], n, p=[0.07, 0.25, 0.48, 0.20])
    distance = np.round(
        np.where(
            area == "Urban", rng.gamma(2, 2, n), np.where(area == "Suburban", rng.gamma(4, 3, n), rng.gamma(6, 5, n))
        ),
        1,
    )
    pool = rng.choice(["Yes", "No"], n, p=[0.15, 0.85])
    listed = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D")

    cond_mult = pd.Series(condition).map({"Poor": 0.82, "Fair": 0.93, "Good": 1.0, "Excellent": 1.12}).to_numpy()
    price = (
        (
            60_000
            + 165 * sqft
            + 9_000 * bathrooms
            + 14_000 * garage
            + 900 * (year_built - 1950)
            + 40_000 * lot
            - 2_200 * distance
            + 25_000 * (pool == "Yes")
        )
        * pd.Series(city).map(cities).to_numpy()
        * cond_mult
        * rng.lognormal(0, 0.08, n)
    )
    price = (np.round(price / 500) * 500).astype(int)

    df = pd.DataFrame(
        {
            "listing_id": np.arange(50001, 50001 + n),
            "listed_date": listed.strftime("%Y-%m-%d"),
            "city": city,
            "area_type": area,
            "sqft": sqft.astype(int),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "lot_size_acres": lot,
            "garage_spaces": garage,
            "condition": condition,
            "distance_to_center_km": distance,
            "has_pool": pool,
            "price": price,
        }
    )
    df["lot_size_acres"] = _with_missing(rng, df["lot_size_acres"], 0.05)
    df["year_built"] = _with_missing(rng, df["year_built"].astype(float), 0.02).astype("Int64")
    return df


def retail_orders(rng: np.random.Generator, n: int = 3000) -> pd.DataFrame:
    categories = {"Electronics": 220, "Apparel": 45, "Home": 70, "Beauty": 25, "Sports": 60}
    category = rng.choice(list(categories), n, p=[0.2, 0.3, 0.2, 0.15, 0.15])
    days = rng.integers(0, 730, n)
    seasonal = 1 + 0.35 * np.isin((pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")).month, [11, 12])
    order_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, unit="D")
    region = rng.choice(["North America", "Europe", "Asia Pacific", "Latin America"], n, p=[0.4, 0.3, 0.2, 0.1])
    channel = rng.choice(["Web", "Mobile app", "Marketplace"], n, p=[0.45, 0.35, 0.2])
    segment = rng.choice(["Consumer", "Corporate", "Small business"], n, p=[0.6, 0.25, 0.15])
    units = np.clip(rng.poisson(2.2 * seasonal), 1, 12)
    unit_price = np.round(pd.Series(category).map(categories).to_numpy() * rng.lognormal(0, 0.35, n), 2)
    discount = np.round(
        rng.choice([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4], n, p=[0.35, 0.15, 0.18, 0.12, 0.1, 0.06, 0.04]), 2
    )
    revenue = np.round(units * unit_price * (1 - discount), 2)
    shipping = np.clip(rng.poisson(np.where(region == "North America", 3, 5)), 1, 14)
    rating = np.clip(np.round(4.3 - 0.15 * shipping + rng.normal(0, 0.8, n)), 1, 5)
    logit = (
        -3.6
        + 2.0 * (category == "Apparel")
        + 1.0 * (category == "Electronics")
        + 5.0 * discount
        + 0.3 * shipping
        - 0.9 * (rating - 3)
        + 0.8 * (channel == "Marketplace")
    )
    returned = np.where(rng.random(n) < _sigmoid(logit), "Yes", "No")

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{2300000 + i}" for i in range(n)],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "region": region,
            "channel": channel,
            "customer_segment": segment,
            "category": category,
            "units": units,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "shipping_days": shipping,
            "rating": rating,
            "returned": returned,
        }
    )
    df["rating"] = _with_missing(rng, df["rating"], 0.22)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    builders = {"customer_churn": customer_churn, "house_prices": house_prices, "retail_orders": retail_orders}
    for name, build in builders.items():
        df = build(np.random.default_rng(7))
        path = OUT / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path.relative_to(OUT.parent.parent)}  ({len(df):,} rows × {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
