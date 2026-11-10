"""
Quick test runner for safety.py — no pytest needed.
Run: python run_tests.py
"""
import sys, os
sys.path.insert(0, ".")

from agent.safety import validate_sql, enforce_limit, safe_query, SafetyError
from agent.ingest import clean_column_name, ingest_dataset

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: unexpected error: {e}")
        failed += 1

def expect_no_raise(sql):
    validate_sql(sql)

def expect_raises(sql, label="safety error"):
    try:
        validate_sql(sql)
        raise AssertionError(f"Expected SafetyError for: {sql[:60]}")
    except SafetyError:
        pass  # expected

print("\n=== Safety Layer Tests ===\n")

# -- SELECT passes --
test("simple SELECT passes",
     lambda: expect_no_raise("SELECT * FROM sales"))
test("SELECT with WHERE passes",
     lambda: expect_no_raise("SELECT region, SUM(sales) FROM sales WHERE region='West' GROUP BY region"))
test("SELECT CTE passes",
     lambda: expect_no_raise("WITH t AS (SELECT region FROM sales) SELECT * FROM t"))
test("lowercase select passes",
     lambda: expect_no_raise("select * from sales limit 10"))

# -- DML blocked --
test("INSERT blocked",
     lambda: expect_raises("INSERT INTO sales (region) VALUES ('X')"))
test("UPDATE blocked",
     lambda: expect_raises("UPDATE sales SET sales=0"))
test("DELETE blocked",
     lambda: expect_raises("DELETE FROM sales"))

# -- DDL blocked --
test("DROP blocked",
     lambda: expect_raises("DROP TABLE sales"))
test("CREATE blocked",
     lambda: expect_raises("CREATE TABLE evil (x TEXT)"))
test("ALTER blocked",
     lambda: expect_raises("ALTER TABLE sales ADD COLUMN x TEXT"))

# -- Multi-statement injection --
test("SELECT; DROP blocked",
     lambda: expect_raises("SELECT 1; DROP TABLE sales"))
test("SELECT; DELETE blocked",
     lambda: expect_raises("SELECT * FROM sales; DELETE FROM sales"))

# -- Empty query --
test("empty query blocked",
     lambda: expect_raises(""))

# -- enforce_limit --
def test_adds_limit():
    r = enforce_limit("SELECT * FROM sales", max_rows=500)
    assert "LIMIT 500" in r.upper(), f"Expected LIMIT 500, got: {r}"

def test_caps_limit():
    r = enforce_limit("SELECT * FROM sales LIMIT 10000", max_rows=500)
    assert "LIMIT 500" in r.upper(), f"Expected LIMIT 500, got: {r}"
    assert "10000" not in r

def test_keeps_small_limit():
    r = enforce_limit("SELECT * FROM sales LIMIT 50", max_rows=500)
    assert "LIMIT 50" in r, f"Expected LIMIT 50, got: {r}"

test("enforce_limit adds LIMIT when absent", test_adds_limit)
test("enforce_limit caps large LIMIT",        test_caps_limit)
test("enforce_limit keeps small LIMIT",       test_keeps_small_limit)

# -- safe_query pipeline --
def test_safe_query_ok():
    r = safe_query("SELECT region FROM sales GROUP BY region")
    assert "LIMIT" in r.upper()

def test_safe_query_blocks_delete():
    try:
        safe_query("DELETE FROM sales")
        raise AssertionError("Should have raised SafetyError")
    except SafetyError:
        pass

test("safe_query returns query with LIMIT",   test_safe_query_ok)
test("safe_query blocks DELETE",              test_safe_query_blocks_delete)

# -- Dynamic Dataset Ingestion --
def test_ingest_cleanup():
    assert clean_column_name("Customer Name ($)") == "customer_name"
    assert clean_column_name("2024 Revenue") == "col_2024_revenue"

def test_ingest_csv_run():
    import io
    csv_data = io.BytesIO(b"Item,Price,Qty\nWidget,12.50,100\nGadget,45.00,20\n")
    csv_data.name = "inventory.csv"
    res = ingest_dataset(csv_data, table_name="test_inventory", db_path="data/test_sales.db")
    assert res["row_count"] == 2
    assert "item" in res["columns"]
    if os.path.exists("data/test_sales.db"):
        os.remove("data/test_sales.db")

test("clean_column_name sanitizes headers",   test_ingest_cleanup)
test("ingest_dataset loads CSV into SQLite",  test_ingest_csv_run)

print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*40}")
sys.exit(0 if failed == 0 else 1)
