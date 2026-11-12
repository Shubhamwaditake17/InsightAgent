import sqlite3
import numpy as np
import pandas as pd

DATA_DICTIONARY = [
    "sales table: one row per region, product and month.",
    "region: sales region (North, South, East, West).",
    "product: product line (A, B, C).",
    "month: calendar month as YYYY-MM.",
    "units: number of units sold in that month.",
    "price: unit price in USD.",
    "revenue: units multiplied by price in USD.",
]


def build_dataframe(seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for region in ["North", "South", "East", "West"]:
        for product, base_price in [("A", 20.0), ("B", 35.0), ("C", 50.0)]:
            for m in range(1, 13):
                units = int(rng.normal(200, 25))
                if region == "West" and m >= 10:   # built-in Q4 drop for analysis demos
                    units = int(units * 0.6)
                rows.append((region, product, f"2025-{m:02d}", units, base_price))
    df = pd.DataFrame(rows, columns=["region", "product", "month", "units", "price"])
    df = df.drop_duplicates().dropna()
    df = df[df.units > 0]
    df["revenue"] = (df.units * df.price).round(2)
    return df


def build_db(path="data/sales.db"):
    df = build_dataframe()
    con = sqlite3.connect(path)
    df.to_sql("sales", con, if_exists="replace", index=False)
    con.close()
    return len(df)


if __name__ == "__main__":
    print("rows loaded:", build_db())
