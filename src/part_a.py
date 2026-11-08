from pathlib import Path
import pandas as pd

# Project and data folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Create processed folder if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load customer data
customers = pd.read_csv(RAW_DIR / "customers.csv")

print("Original data loaded successfully!")
print("Original customer count:", len(customers))


# ==========================================
# CLEAN CUSTOMERS DATA
# ==========================================

# Clean city names
customers["city"] = (
    customers["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

# Clean gender values
customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({
        "f": "female",
        "m": "male",
        "": pd.NA
    })
)

customers["gender"] = customers["gender"].fillna("unknown")

# Clean age values
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# Ages outside the inclusive 10-100 range are invalid.
invalid_age_mask = customers["age"].notna() & ~customers["age"].between(10, 100)
invalid_age_count = int(invalid_age_mask.sum())
customers.loc[invalid_age_mask, "age"] = pd.NA
missing_age_before_fill = int(customers["age"].isna().sum())

# Fill missing age with the median age for the same city (groupby + transform).
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)
print(f"Invalid ages outside 10-100 set to missing: {invalid_age_count}")
print(f"Missing ages filled with city median: {missing_age_before_fill}")

# Convert signup date to a consistent date format
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Show cleaned results
print("\nCity values after cleaning:")
print(customers["city"].value_counts(dropna=False))

print("\nGender values after cleaning:")
print(customers["gender"].value_counts(dropna=False))

print("\nMissing values after cleaning:")
print(customers.isnull().sum())

# Save cleaned customer data in the required folder
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

output_file = CLEAN_DIR / "customers.csv"
customers.to_csv(output_file, index=False)

print("\nCustomers cleaned and saved successfully!")
print("Output file:", output_file)

# Inspect other raw CSV files
print("\n========== OTHER DATA FILES ==========")

files = [
    "products.csv",
    "orders.csv",
    "tickets.csv",
    "policy_eval.csv",
    "agent_scenarios.csv"
]

for file_name in files:
    file_path = RAW_DIR / file_name

    if file_path.exists():
        df = pd.read_csv(file_path)

        print(f"\nFile: {file_name}")
        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())
        print("Missing values:", df.isnull().sum().sum())
    else:
        print(f"\nFile not found: {file_name}")

# Inspect missing values in orders
print("\n========== ORDERS DATA QUALITY ==========")

orders = pd.read_csv(RAW_DIR / "orders.csv")

print("\nMissing values and percentage in each column:")

missing_summary = pd.DataFrame({
    "Missing_Count": orders.isnull().sum(),
    "Missing_Percentage": (orders.isnull().mean() * 100).round(2)
})

print(missing_summary)

print("\nFirst 5 order records:")
print(orders.head())

print("\nOrder status values:")
print(orders["status"].value_counts(dropna=False))

print("\nOrder data types:")
print(orders.dtypes)
# ==========================================
# CLEAN ORDERS DATA
# ==========================================

print("\n========== CLEANING ORDERS DATA ==========")

# Load original orders and products
orders = pd.read_csv(RAW_DIR / "orders.csv")
products = pd.read_csv(RAW_DIR / "products.csv")

# Convert columns to correct data types
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

for column in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
    orders[column] = pd.to_numeric(
        orders[column], errors="coerce"
    )

orders["status"] = (
    orders["status"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Step 1: Remove completely identical rows.
before = len(orders)
orders = orders.drop_duplicates()
exact_duplicates_removed = before - len(orders)
print("Exact duplicate rows removed:", exact_duplicates_removed)

# Step 2: Keep only the latest record for each order_id.
before = len(orders)
orders = orders.sort_values("order_ts").drop_duplicates(
    subset="order_id", keep="last"
)
order_id_rows_removed = before - len(orders)
print("Older duplicate order records removed:", order_id_rows_removed)
print("Orders after keeping latest record:", len(orders))

# Step 3: Convert amounts to numeric and fill unparseable/missing amounts.
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

amounts_filled = int(orders["order_amount"].isna().sum())
orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)
print("Missing/unparseable order amounts filled from product price:", amounts_filled)

# Step 4: Make negative delivery times missing.
negative_delivery_count = int(orders["delivery_minutes"].lt(0).sum())
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA
print("Negative delivery times set to missing:", negative_delivery_count)

# Step 5: Fill missing delivery times only for completed orders, using the
# completed-order median for each product category.
completed_mask = orders["status"] == "completed"
completed_missing_before_fill = int(orders.loc[completed_mask, "delivery_minutes"].isna().sum())

category_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"].fillna(category_median)
)
print("Missing completed-order delivery times filled:", completed_missing_before_fill)
print("Cancelled-order delivery times left missing:", int(orders.loc[~completed_mask, "delivery_minutes"].isna().sum()))

# Remove helper columns.
orders = orders.drop(columns=["price_inr", "category"])

# Step 6: Save cleaned orders
orders = orders.sort_values("order_id")

CLEAN_DIR = PROJECT_DIR / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

orders_output = CLEAN_DIR / "orders.csv"
orders.to_csv(orders_output, index=False)

print("\nOrders cleaned and saved successfully!")
print("Output file:", orders_output)

# Step 7: Check final results
print("\n========== FINAL ORDER CHECKS ==========")

print("Total orders:", len(orders))

print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nMissing values:")
print(orders.isnull().sum())

print("\nDuplicate order IDs:", orders["order_id"].duplicated().sum())

print(
    "\nCompleted order amount total:",
    orders.loc[
        orders["status"] == "completed", "order_amount"
    ].sum().round(2)
)

# ==========================================
# INVESTIGATE DUPLICATE ORDERS
# ==========================================

print("\n========== DUPLICATE ORDER INVESTIGATION ==========")

# Find duplicated order IDs
duplicate_rows = orders[
    orders["order_id"].duplicated(keep=False)
].sort_values("order_id")

print("\nNumber of rows with duplicated order IDs:")
print(len(duplicate_rows))

print("\nFirst 20 rows with duplicated order IDs:")
print(duplicate_rows.head(20).to_string(index=False))

# Check whether duplicated rows are completely identical
print("\nCompletely duplicated rows:")
print(orders.duplicated().sum())

# Check order IDs with the highest repetition
print("\nMost repeated order IDs:")
print(orders["order_id"].value_counts().head(10))

print("\n========== CLEANING TICKETS DATA ==========")

# Load tickets data
tickets = pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False)

# Display original information
print("\nOriginal tickets count:", len(tickets))

print("\nMissing values before cleaning:")
print(tickets.isnull().sum())

# Clean text columns
for column in ["ticket_id", "customer_id", "order_id", "text", "true_category"]:
    tickets[column] = tickets[column].astype("string").str.strip()

# Keep empty ticket text as an empty string. Part C uses this distinction
# to identify and skip the four intentionally empty tickets.

# Check duplicate ticket IDs
print("\nDuplicate ticket IDs:", tickets["ticket_id"].duplicated().sum())

# Remove completely identical rows only
before_count = len(tickets)
tickets = tickets.drop_duplicates()

print("Completely identical rows removed:", before_count - len(tickets))

# Display missing values after cleaning
print("\nMissing values after cleaning:")
print(tickets.isnull().sum())

# Save cleaned tickets
tickets.to_csv(OUTPUT_DIR / "tickets_cleaned.csv", index=False)

print("\nTickets cleaned and saved successfully!")
print("Output file:", OUTPUT_DIR / "tickets_cleaned.csv")

print("\n========== CLEANING PRODUCTS DATA ==========")

# Load products data
products = pd.read_csv(RAW_DIR / "products.csv")

print("\nOriginal products count:", len(products))

print("\nMissing values before cleaning:")
print(products.isnull().sum())

# Clean text columns
for column in ["product_id", "product_name", "category"]:
    products[column] = products[column].astype("string").str.strip()

# Convert price to numeric
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# Check duplicate product IDs
print(
    "\nDuplicate product IDs:",
    products["product_id"].duplicated().sum()
)

# Remove completely identical rows
before_count = len(products)
products = products.drop_duplicates()

print(
    "Completely identical rows removed:",
    before_count - len(products)
)

print("\nMissing values after cleaning:")
print(products.isnull().sum())

# Save cleaned products in the required folder
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

products_output = CLEAN_DIR / "products.csv"
products.to_csv(products_output, index=False)

print("\nProducts cleaned and saved successfully!")
print("Output file:", products_output)

print("\n========== CHECKING POLICY EVALUATION DATA ==========")

# Load policy evaluation data
policy_eval = pd.read_csv(RAW_DIR / "policy_eval.csv")

print("\nTotal policy evaluation records:", len(policy_eval))

print("\nFirst 5 records:")
print(policy_eval.head().to_string(index=False))

print("\nMissing values:")
print(policy_eval.isnull().sum())

print("\nDuplicate rows:", policy_eval.duplicated().sum())

# Clean text fields without inventing missing information
for column in ["question", "expected_doc", "expected_fact"]:
    policy_eval[column] = policy_eval[column].astype("string").str.strip()

policy_eval = policy_eval.replace(r"^\s*$", pd.NA, regex=True)

# Save cleaned data
policy_eval.to_csv(
    OUTPUT_DIR / "policy_eval_cleaned.csv",
    index=False
)

print("\nPolicy evaluation data saved successfully!")
print("Output file:", OUTPUT_DIR / "policy_eval_cleaned.csv")

print("\nMissing values after cleaning:")
print(policy_eval.isnull().sum())

print("\n========== CHECKING AGENT SCENARIOS DATA ==========")

# Load agent scenarios
scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv")

print("\nTotal scenarios:", len(scenarios))

print("\nFirst 5 scenarios:")
print(scenarios.head().to_string(index=False))

print("\nMissing values before cleaning:")
print(scenarios.isnull().sum())

print("\nDuplicate scenario IDs:")
print(scenarios["scenario_id"].duplicated().sum())

# Clean text columns
for column in ["scenario_id", "customer_id", "message", "expected_outcome"]:
    scenarios[column] = scenarios[column].astype("string").str.strip()

# Convert expected refund to numeric
scenarios["expected_refund_inr"] = pd.to_numeric(
    scenarios["expected_refund_inr"], errors="coerce"
)

# Convert empty strings to missing values
scenarios = scenarios.replace(r"^\s*$", pd.NA, regex=True)

# Remove only completely identical rows
before_count = len(scenarios)
scenarios = scenarios.drop_duplicates()

print("\nCompletely identical rows removed:", before_count - len(scenarios))

print("\nMissing values after cleaning:")
print(scenarios.isnull().sum())

# Save cleaned scenarios
scenarios.to_csv(
    OUTPUT_DIR / "agent_scenarios_cleaned.csv",
    index=False
)

print("\nAgent scenarios saved successfully!")
print("Output file:", OUTPUT_DIR / "agent_scenarios_cleaned.csv")

# Create SQLite database
import sqlite3

CLEAN_DIR = Path("data/clean")
DB_PATH = Path("data/instamart.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

conn = sqlite3.connect(DB_PATH)

customers.to_sql("customers", conn, if_exists="replace", index=False)
products.to_sql("products", conn, if_exists="replace", index=False)
orders.to_sql("orders", conn, if_exists="replace", index=False)

conn.close()

print("\nSQLite database created successfully!")
print("Database location:", DB_PATH)
print("Tables created: customers, products, orders")