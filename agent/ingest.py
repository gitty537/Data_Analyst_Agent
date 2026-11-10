"""
agent/ingest.py
---------------
Dynamic dataset ingestion into SQLite.

Allows users to upload any CSV or Excel file, automatically clean its column
names for reliable SQL execution, and register it inside SQLite as an active
table that the agent can immediately query.
"""

import io
import re
import sqlite3
import pandas as pd
from typing import BinaryIO, Union

DB_PATH = "data/sales.db"


def clean_column_name(col: str) -> str:
    """
    Sanitize column names for clean SQL queries:
    - Lowercase
    - Replace whitespace and special characters with underscores
    - Remove consecutive underscores
    - Avoid starting with a number
    """
    col = str(col).strip().lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    if not col:
        col = "col"
    if col[0].isdigit():
        col = f"col_{col}"
    return col


def ingest_dataset(
    file_or_path: Union[str, BinaryIO, io.BytesIO],
    table_name: str = "uploaded_dataset",
    db_path: str = DB_PATH,
) -> dict:
    """
    Ingest a CSV or Excel file into SQLite.

    Returns:
      dict with keys:
        - "table_name": str
        - "columns": list[str]
        - "row_count": int
        - "sample": list[dict]
    """
    # 1. Read file into DataFrame
    if isinstance(file_or_path, str):
        if file_or_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_or_path)
        else:
            df = pd.read_csv(file_or_path)
    else:
        # Streamlit UploadedFile or BytesIO
        file_name = getattr(file_or_path, "name", "uploaded.csv").lower()
        if file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_or_path)
        else:
            df = pd.read_csv(file_or_path)

    if df.empty:
        raise ValueError("The uploaded file contains no data.")

    # 2. Sanitize column names
    clean_cols = [clean_column_name(c) for c in df.columns]

    # Handle duplicate column names if any
    seen = {}
    deduped = []
    for c in clean_cols:
        if c in seen:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            deduped.append(c)

    df.columns = deduped

    # 3. Write to SQLite
    table_name = clean_column_name(table_name)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    return {
        "table_name": table_name,
        "columns": deduped,
        "row_count": len(df),
        "sample": df.head(5).to_dict(orient="records"),
    }
