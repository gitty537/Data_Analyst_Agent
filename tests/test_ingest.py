"""
tests/test_ingest.py
--------------------
Unit tests for agent/ingest.py (Dataset Upload & Dynamic Ingestion).
"""

import os
import sqlite3
import pandas as pd
import pytest
from agent.ingest import clean_column_name, ingest_dataset

TEST_DB = "data/test_sales.db"


def test_clean_column_name():
    assert clean_column_name("Order ID") == "order_id"
    assert clean_column_name("Sales ($)") == "sales"
    assert clean_column_name("  Customer - Name  ") == "customer_name"
    assert clean_column_name("2024 Revenue") == "col_2024_revenue"
    assert clean_column_name("") == "col"


def test_ingest_dataset_csv(tmp_path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("Product Name,Unit Price,Stock Qty\nLaptop,999.50,15\nMouse,25.00,100\n", encoding="utf-8")

    result = ingest_dataset(str(csv_file), table_name="test_inventory", db_path=TEST_DB)
    assert result["table_name"] == "test_inventory"
    assert result["row_count"] == 2
    assert "product_name" in result["columns"]
    assert "unit_price" in result["columns"]
    assert "stock_qty" in result["columns"]

    # Verify rows exist in DB
    conn = sqlite3.connect(TEST_DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM test_inventory")
    count = cur.fetchone()[0]
    conn.close()

    assert count == 2

    # Clean up test DB
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
