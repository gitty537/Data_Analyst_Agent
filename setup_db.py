"""
setup_db.py
-----------
Reads data/sales.csv and loads it into data/sales.db (SQLite).

WHY SQLite?
  - Zero-install: ships with Python's standard library.
  - File-based: easy to version-control or share.
  - SQL-compatible: lets us demonstrate structural read-only enforcement.
  - Fast enough for 10 k rows with no extra config.

WHY a separate setup script instead of doing it in the agent?
  - Separation of concerns: data ingestion happens once, the agent only reads.
  - Makes it trivial to swap in a different data source later.
"""

import csv
import sqlite3
import os

CSV_PATH = "data/sales.csv"
DB_PATH  = "data/sales.db"


def infer_type(value: str) -> str:
    """
    Guess the SQLite column type from a sample string value.
    SQLite is loosely typed, but declaring types helps query optimisers and
    makes the schema description clearer for the LLM.
    """
    # Try integer first
    try:
        int(value)
        return "INTEGER"
    except ValueError:
        pass
    # Then float
    try:
        float(value)
        return "REAL"
    except ValueError:
        pass
    # Anything else is text
    return "TEXT"


def load_csv_to_sqlite(csv_path: str = CSV_PATH, db_path: str = DB_PATH):
    """
    1. Read the CSV header to get column names.
    2. Peek at the first data row to infer column types.
    3. CREATE TABLE IF NOT EXISTS with those types.
    4. INSERT all rows using executemany() for speed.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found — run `python generate_data.py` first."
        )

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)           # read all rows into memory (safe for 10 k)

    if not rows:
        raise ValueError("CSV file is empty.")

    # Peek at first row to infer types
    sample   = rows[0]
    columns  = list(sample.keys())
    col_types = {col: infer_type(sample[col]) for col in columns}

    # Build CREATE TABLE statement
    # We quote column names with double quotes to handle names with spaces safely.
    col_defs = ", ".join(f'"{c}" {col_types[c]}' for c in columns)
    create_sql = f'CREATE TABLE IF NOT EXISTS sales ({col_defs});'

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # Drop and recreate so re-running this script is idempotent
    cur.execute("DROP TABLE IF EXISTS sales;")
    cur.execute(create_sql)

    # Build INSERT statement with named placeholders (safe against SQL injection
    # even though we control the data here — good practice)
    placeholders = ", ".join(f":{c}" for c in columns)
    insert_sql   = f"INSERT INTO sales VALUES ({placeholders});"

    # executemany is much faster than looping individual inserts because it
    # batches the work in a single transaction.
    cur.executemany(insert_sql, rows)
    conn.commit()
    conn.close()

    print(f"[OK] Loaded {len(rows)} rows into {db_path}")
    print(f"     Table: sales | Columns: {', '.join(columns)}")


if __name__ == "__main__":
    load_csv_to_sqlite()
