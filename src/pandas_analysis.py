
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

ORDERS_FILE = PROJECT_DIR / "data" / "clean" / "orders.csv"
CUSTOMERS_FILE = PROJECT_DIR / "data" / "clean" / "customers.csv"

PRODUCTS_FILE = PROJECT_DIR / "data" / "clean" / "products.csv"

orders = pd.read_csv(ORDERS_FILE)
customers = pd.read_csv(CUSTOMERS_FILE)

products = pd.read_csv(PRODUCTS_FILE)

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Keep completed orders
completed = orders[orders["status"] == "completed"].copy()

# Merge orders with customers
merged = completed.merge(
    customers,
    on="customer_id",
    how="left"
)

# Merge products to get category
merged = merged.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# City-wise revenue analysis
city_analysis = merged.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    total_orders=("order_id", "count")
).round(2)

print("\n===== CITY-WISE ANALYSIS =====")
print(city_analysis)

# Overall order amount statistics
print("\n===== ORDER AMOUNT STATISTICS =====")
print(completed["order_amount"].describe())

# Median and skewness
print("\nMedian:", completed["order_amount"].median())
print("Skewness:", completed["order_amount"].skew())

# Correlation
print("\n===== CORRELATION =====")
print(
    completed[
        [
            "order_amount",
            "quantity",
            "discount_pct",
            "delivery_minutes"
        ]
    ]
    .corr()
    .round(3)
)


# ===== OUTLIER ANALYSIS USING IQR =====

Q1 = completed["order_amount"].quantile(0.25)
Q3 = completed["order_amount"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = completed[
    (completed["order_amount"] < lower_limit) |
    (completed["order_amount"] > upper_limit)
]

print("\n===== OUTLIER ANALYSIS =====")
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower limit:", round(lower_limit, 2))
print("Upper limit:", round(upper_limit, 2))
print("Total outliers:", len(outliers))

print("Outliers below lower limit:",
      len(completed[completed["order_amount"] < lower_limit]))

print("Outliers above upper limit:",
      len(completed[completed["order_amount"] > upper_limit]))


# ===== PIVOT TABLE ANALYSIS =====

pivot = pd.pivot_table(
    merged,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All",
)

print("\n===== CITY AND CATEGORY PIVOT TABLE (WITH MARGINS) =====")
print(pivot.round(2))

pivot_grand_total = round(float(pivot.loc["All", "All"]), 2)
print("\n===== PIVOT TABLE GRAND TOTAL =====")
print(pivot_grand_total)
assert pivot_grand_total == 2998668.80

print("\n===== CITY-WISE NAMED AGGREGATION =====")
city_named = merged.groupby("city").agg(
    revenue_sum=("order_amount", "sum"),
    revenue_mean=("order_amount", "mean"),
    completed_order_count=("order_id", "count"),
)
print(city_named.round({"revenue_sum": 2, "revenue_mean": 2}))
