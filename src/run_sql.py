"""Execute the seven required Part A SQL queries against the SQLite database."""
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

sql_text = SQL_PATH.read_text(encoding="utf-8")
queries = [q.strip() for q in sql_text.split(";") if q.strip()]
if len(queries) != 7:
    raise RuntimeError(f"Expected exactly 7 SQL queries, found {len(queries)}")

expected_rows = [30, 30, 11, 25, 550, 1175, 183]
with sqlite3.connect(DB_PATH) as conn:
    for number, (query, expected) in enumerate(zip(queries, expected_rows), start=1):
        result = pd.read_sql_query(query, conn)
        print(f"\n========== Q{number} ==========")
        print(result.to_string(index=False))
        print(f"Rows: {len(result)} (expected {expected})")
        if len(result) != expected:
            raise AssertionError(f"Q{number}: expected {expected} rows, got {len(result)}")
        if number == 2:
            total = round(result["total_revenue"].sum(), 2)
            max_row = result.loc[result["total_revenue"].idxmax()]
            assert total == 2998668.80, total
            assert (max_row["city"], max_row["category"], round(max_row["total_revenue"], 2)) == ("Bengaluru", "Electronics", 233909.45)
            print(f"Grand total: INR {total:,.2f}")
            print(f"Largest cell: {max_row['city']} / {max_row['category']} = INR {max_row['total_revenue']:,.2f}")
        elif number == 5:
            avg = pd.read_sql_query("SELECT AVG(total_spend) AS avg_spend FROM (SELECT customer_id, SUM(order_amount) AS total_spend FROM orders WHERE status='completed' GROUP BY customer_id)", conn).iloc[0, 0]
            print(f"Average completed spend: INR {avg:,.2f}")
        elif number == 6:
            diff = int((result["city_rank"] != result["row_num"]).sum())
            print(f"Rows where RANK differs from ROW_NUMBER: {diff}")
            assert diff == 8, diff
        elif number == 7:
            total = round(result["cumulative_revenue"].iloc[-1], 2)
            print(f"Final cumulative revenue: INR {total:,.2f}")
            assert total == 2998668.80, total
print("\nALL 7 REQUIRED SQL QUERIES PASSED")
