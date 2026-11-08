"""Audit raw Instamart CSV exports without modifying the source files."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# Preserve empty ticket text as an empty string, as required by the brief.
tables = {
    "customers": pd.read_csv(RAW / "customers.csv"),
    "products": pd.read_csv(RAW / "products.csv"),
    "orders": pd.read_csv(RAW / "orders.csv"),
    "tickets": pd.read_csv(RAW / "tickets.csv", keep_default_na=False),
}

print("===== RAW TABLE AUDIT =====")
for name, df in tables.items():
    print(f"\n[{name}] shape={df.shape}")
    print("Data types:")
    print(df.dtypes.to_string())
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    # For tickets, empty strings are intentionally not NA; report them separately.
    if name == "tickets":
        empty_text = df["text"].astype(str).str.strip().eq("").sum()
        blank_order_ids = df["order_id"].astype(str).str.strip().eq("").sum()
        print(f"empty_text_count={empty_text}")
        print(f"blank_order_id_count={blank_order_ids}")
    print("Missing values by column:")
    print(missing.to_string())

orders = tables["orders"]
customers = tables["customers"]
amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery_numeric = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
ages = pd.to_numeric(customers["age"], errors="coerce")

figures = {
    "raw_order_rows": len(orders),
    "exact_duplicate_rows": int(orders.duplicated().sum()),
    "pending_rows": int(orders["status"].astype(str).str.lower().eq("pending").sum()),
    "distinct_order_ids": int(orders["order_id"].nunique()),
    "non_numeric_order_amounts": int(amount_numeric.isna().sum()),
    "negative_delivery_minutes": int(delivery_numeric.lt(0).sum()),
    "blank_delivery_minutes": int(orders["delivery_minutes"].isna().sum() + orders["delivery_minutes"].astype(str).str.strip().eq("").sum()),
    "blank_customer_ages": int(customers["age"].isna().sum() + customers["age"].astype(str).str.strip().eq("").sum()),
    "ages_outside_10_100": int((ages.lt(10) | ages.gt(100)).sum()),
    "distinct_raw_city_spellings": int(customers["city"].nunique(dropna=True)),
    "distinct_nonblank_gender_spellings": int(customers.loc[customers["gender"].astype(str).str.strip().ne(""), "gender"].nunique()),
    "blank_genders": int(customers["gender"].isna().sum() + customers["gender"].astype(str).str.strip().eq("").sum()),
}

print("\n===== REQUIRED RAW AUDIT FIGURES =====")
for key, value in figures.items():
    print(f"{key}: {value}")

expected = {
    "raw_order_rows": 13084,
    "exact_duplicate_rows": 119,
    "pending_rows": 656,
    "distinct_order_ids": 12309,
    "non_numeric_order_amounts": 120,
    "negative_delivery_minutes": 106,
    "blank_delivery_minutes": 2541,
    "blank_customer_ages": 61,
    "ages_outside_10_100": 6,
    "distinct_raw_city_spellings": 24,
    "distinct_nonblank_gender_spellings": 10,
    "blank_genders": 30,
}
failed = {k: (figures[k], v) for k, v in expected.items() if figures[k] != v}
if failed:
    raise SystemExit(f"RAW AUDIT CHECKPOINT MISMATCH: {failed}")
print("\nALL RAW AUDIT REFERENCE FIGURES MATCH")
