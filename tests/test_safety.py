"""
tests/test_safety.py
--------------------
Unit tests for agent/safety.py — the structural read-only SQL enforcement layer.

WHY unit test the safety layer specifically?
  It's the most security-critical component.  If this breaks, a malicious or
  hallucinating LLM could wipe the database.  Tests make it regression-proof.

Run:
  python -m pytest tests/test_safety.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.safety import validate_sql, enforce_limit, safe_query, SafetyError


# ===========================================================================
# Tests for validate_sql()
# ===========================================================================

class TestValidateSql:
    """Tests that validate_sql() correctly allows/blocks statements."""

    # ── Allowed: SELECT variants ──────────────────────────────────────────────

    def test_simple_select_passes(self):
        """The most basic valid query must pass."""
        validate_sql("SELECT * FROM sales")  # should not raise

    def test_select_with_where_passes(self):
        validate_sql("SELECT region, SUM(sales) FROM sales WHERE year = 2023 GROUP BY region")

    def test_select_with_subquery_passes(self):
        validate_sql("SELECT * FROM (SELECT region, SUM(sales) FROM sales GROUP BY region) t")

    def test_select_with_cte_passes(self):
        """Common Table Expressions (WITH ... AS) must pass."""
        validate_sql("""
            WITH agg AS (SELECT region, SUM(sales) AS s FROM sales GROUP BY region)
            SELECT * FROM agg ORDER BY s DESC
        """)

    def test_select_case_insensitive_passes(self):
        """SQL is case-insensitive; lowercase 'select' must pass."""
        validate_sql("select * from sales limit 10")

    # ── Blocked: DML write statements ─────────────────────────────────────────

    def test_insert_blocked(self):
        with pytest.raises(SafetyError, match="Only SELECT"):
            validate_sql("INSERT INTO sales (region) VALUES ('West')")

    def test_update_blocked(self):
        with pytest.raises(SafetyError, match="Only SELECT"):
            validate_sql("UPDATE sales SET sales = 0")

    def test_delete_blocked(self):
        with pytest.raises(SafetyError, match="Only SELECT"):
            validate_sql("DELETE FROM sales WHERE region = 'West'")

    # ── Blocked: DDL statements ───────────────────────────────────────────────

    def test_drop_blocked(self):
        with pytest.raises(SafetyError):
            validate_sql("DROP TABLE sales")

    def test_create_blocked(self):
        with pytest.raises(SafetyError):
            validate_sql("CREATE TABLE evil (x TEXT)")

    def test_alter_blocked(self):
        with pytest.raises(SafetyError):
            validate_sql("ALTER TABLE sales ADD COLUMN extra TEXT")

    # ── Blocked: Multi-statement injections ───────────────────────────────────

    def test_select_then_drop_blocked(self):
        """Classic SQL injection: a SELECT followed by a destructive statement."""
        with pytest.raises(SafetyError):
            validate_sql("SELECT 1; DROP TABLE sales")

    def test_select_then_delete_blocked(self):
        with pytest.raises(SafetyError):
            validate_sql("SELECT * FROM sales; DELETE FROM sales")

    # ── Blocked: Embedded banned keywords ────────────────────────────────────

    def test_embedded_drop_in_select_blocked(self):
        """
        SELECT with a DROP keyword somewhere inside must be caught by Layer 2.
        In practice this can't be valid SQL, but we test defensively.
        """
        with pytest.raises(SafetyError):
            validate_sql("SELECT * FROM sales WHERE region = 'DROP'")
        # Note: this WILL be caught because 'DROP' is a banned keyword anywhere.
        # If you have a region literally called 'DROP', the LLM should use
        # a parameterised query or alias. This is an intentional conservative choice.

    def test_empty_query_blocked(self):
        with pytest.raises(SafetyError, match="Empty"):
            validate_sql("")

    def test_whitespace_only_blocked(self):
        with pytest.raises(SafetyError, match="Empty"):
            validate_sql("   \n\t  ")


# ===========================================================================
# Tests for enforce_limit()
# ===========================================================================

class TestEnforceLimit:
    """Tests that enforce_limit() correctly adds or caps the LIMIT clause."""

    def test_adds_limit_when_absent(self):
        result = enforce_limit("SELECT * FROM sales", max_rows=500)
        assert "LIMIT 500" in result.upper()

    def test_does_not_add_duplicate_limit(self):
        result = enforce_limit("SELECT * FROM sales LIMIT 100", max_rows=500)
        # Should still have exactly one LIMIT
        assert result.upper().count("LIMIT") == 1
        assert "LIMIT 100" in result

    def test_caps_limit_exceeding_max(self):
        result = enforce_limit("SELECT * FROM sales LIMIT 10000", max_rows=500)
        assert "LIMIT 500" in result.upper()
        assert "10000" not in result

    def test_keeps_limit_below_max(self):
        result = enforce_limit("SELECT * FROM sales LIMIT 50", max_rows=500)
        assert "LIMIT 50" in result
        # Should NOT be replaced since 50 < 500
        assert "LIMIT 500" not in result

    def test_strips_trailing_semicolon(self):
        result = enforce_limit("SELECT * FROM sales;", max_rows=500)
        # Semicolons before LIMIT would cause a SQL error
        assert ";" not in result.split("LIMIT")[0]

    def test_case_insensitive_detection(self):
        result = enforce_limit("SELECT * FROM sales limit 100", max_rows=500)
        # Should detect lowercase 'limit' and not add another
        assert result.lower().count("limit") == 1


# ===========================================================================
# Tests for safe_query() (combined pipeline)
# ===========================================================================

class TestSafeQuery:
    """Integration tests for the full safety pipeline."""

    def test_valid_select_returns_safe_query(self):
        result = safe_query("SELECT region, SUM(sales) FROM sales GROUP BY region")
        assert result.startswith("SELECT") or result.startswith("select")
        assert "LIMIT" in result.upper()

    def test_invalid_statement_raises(self):
        with pytest.raises(SafetyError):
            safe_query("DELETE FROM sales")

    def test_safe_query_adds_limit(self):
        result = safe_query("SELECT * FROM sales")
        assert "LIMIT" in result.upper()

    def test_safe_query_caps_large_limit(self):
        result = safe_query("SELECT * FROM sales LIMIT 99999", max_rows=500)
        assert "LIMIT 500" in result.upper()
