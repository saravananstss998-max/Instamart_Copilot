from pathlib import Path
import pandas as pd
import sqlite3

PROJECT_DIR = Path(__file__).resolve().parent.parent

ORDERS_FILE = PROJECT_DIR / "data" / "clean" / "orders.csv"
CUSTOMERS_FILE = PROJECT_DIR / "data" / "clean" / "customers.csv"
PRODUCTS_FILE = PROJECT_DIR / "data" / "clean" / "products.csv"
DB_FILE = PROJECT_DIR / "data" / "instamart.db"

orders = pd.read_csv(ORDERS_FILE)
customers = pd.read_csv(CUSTOMERS_FILE)
products = pd.read_csv(PRODUCTS_FILE)

checks = []

# 1. Unique customer IDs
checks.append((
    "Unique customer IDs",
    customers["customer_id"].is_unique
))

# 2. Unique product IDs
checks.append((
    "Unique product IDs",
    products["product_id"].is_unique
))

# 3. Unique order IDs
checks.append((
    "Unique order IDs",
    orders["order_id"].is_unique
))

# 4. Correct order count
checks.append((
    "Total clean orders = 12309",
    len(orders) == 12309
))

# 5. Completed order count
completed = orders[orders["status"] == "completed"]
checks.append((
    "Completed orders = 11110",
    len(completed) == 11110
))

# 6. Cancelled order count
cancelled = orders[orders["status"] == "cancelled"]
checks.append((
    "Cancelled orders = 1199",
    len(cancelled) == 1199
))

# 7. No pending orders
checks.append((
    "No pending orders",
    not (orders["status"] == "pending").any()
))

# 8. No missing order amounts
checks.append((
    "No missing order amounts",
    orders["order_amount"].notna().all()
))

# 9. No negative order amounts
checks.append((
    "No negative order amounts",
    (orders["order_amount"] >= 0).all()
))

# 10. Valid statuses
valid_statuses = {"completed", "cancelled"}
checks.append((
    "Valid order statuses",
    set(orders["status"].dropna().unique()).issubset(valid_statuses)
))

# 11. Completed orders have delivery times
checks.append((
    "Completed orders have delivery times",
    completed["delivery_minutes"].notna().all()
))

# 12. Cancelled orders have missing delivery times
checks.append((
    "Cancelled orders have missing delivery times",
    cancelled["delivery_minutes"].isna().all()
))

# 13. Customer ages are within 18-60
checks.append((
    "Customer ages are between 18 and 60",
    customers["age"].between(18, 60).all()
))

# 14. No missing customer ages
checks.append((
    "No missing customer ages",
    customers["age"].notna().all()
))

# 15. Valid gender values
valid_genders = {"male", "female", "unknown"}
checks.append((
    "Valid gender values",
    set(customers["gender"].dropna().unique()) == valid_genders
))
# 16. Valid cities
valid_cities = {
    "Mumbai", "Delhi", "Bengaluru",
    "Chennai", "Hyderabad", "Pune"
}
checks.append((
    "Only six standardized cities",
    set(customers["city"].dropna().unique()).issubset(valid_cities)
))

# 17. All order customers exist
checks.append((
    "All order customers exist",
    orders["customer_id"].isin(customers["customer_id"]).all()
))

# 18. All order products exist
checks.append((
    "All order products exist",
    orders["product_id"].isin(products["product_id"]).all()
))

# 19. SQLite database exists
checks.append((
    "SQLite database exists",
    DB_FILE.exists()
))

# 20. Database tables exist and revenue matches
if DB_FILE.exists():
    with sqlite3.connect(DB_FILE) as conn:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        required_tables = {"customers", "products", "orders"}

        checks.append((
            "Required SQLite tables exist",
            required_tables.issubset(tables)
        ))

        if required_tables.issubset(tables):
            db_revenue = conn.execute(
                "SELECT ROUND(SUM(order_amount), 2) "
                "FROM orders WHERE status = 'completed'"
            ).fetchone()[0]

            csv_revenue = round(
                completed["order_amount"].sum(), 2
            )

            checks.append((
                "Database and CSV revenue match",
                db_revenue is not None
                and round(db_revenue, 2) == csv_revenue
            ))

print("\n===== DATA VALIDATION RESULTS =====")

for name, passed in checks:
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

passed_count = sum(passed for _, passed in checks)

print(f"\nPassed: {passed_count}/{len(checks)}")

if all(passed for _, passed in checks):
    print("\nALL CHECKPOINTS PASSED")
else:
    print("\nSOME CHECKPOINTS FAILED")
