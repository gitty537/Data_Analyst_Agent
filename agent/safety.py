"""
agent/safety.py
---------------
Structural read-only SQL enforcement.

The key design principle from the job description:
  "Enforce read-only SQL STRUCTURALLY in code (reject anything but SELECT,
   add a row limit)."

WHY structural, not instructional?
  Instructional = you tell the LLM "only write SELECT queries" in the system
  prompt.  The LLM can hallucinate or be jailbroken into ignoring that.

  Structural = you parse/validate the SQL in Python BEFORE it ever touches the
  database.  No amount of prompt engineering can bypass a Python `if` statement.

This module provides:
  1. `validate_sql(query)` — raises SafetyError for anything non-SELECT.
  2. `enforce_limit(query, max_rows)` — appends LIMIT if absent.
  3. `safe_query(query, max_rows)` — combines both; returns a safe query string.
"""

import re
import sqlparse                     # pip install sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DDL, DML


class SafetyError(Exception):
    """Raised when a query fails the safety check."""
    pass


# ── Banned SQL keywords ───────────────────────────────────────────────────────
# This is a defence-in-depth list.  The primary check is the DML-type check
# below, but we also block these keywords anywhere in the query to catch
# embedded statements like `SELECT 1; DROP TABLE sales`.
_BANNED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|TRUNCATE"
    r"|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|ANALYZE)\b",
    re.IGNORECASE,
)

# Match a LIMIT clause (we need to detect if one already exists)
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def validate_sql(query: str) -> None:
    """
    Raise SafetyError if `query` is not a plain SELECT statement.

    HOW it works (two independent layers):
      Layer 1 — sqlparse token analysis
        sqlparse parses the SQL into tokens.  We look at the first meaningful
        token's type.  For a valid SELECT it must be DML (Data Manipulation
        Language) of value "SELECT".  INSERT/UPDATE/DELETE are also DML, so we
        check the value too.

      Layer 2 — regex blacklist
        Even if Layer 1 passes (e.g. a SELECT that contains a subquery with
        a banned keyword), the regex catches it.

    Why both layers?
      sqlparse is a lenient parser; it can silently ignore syntax errors.
      The regex is a simple backstop.
    """
    query = query.strip()

    if not query:
        raise SafetyError("Empty query.")

    # --- Layer 1: sqlparse token check ---
    parsed = sqlparse.parse(query)
    if not parsed:
        raise SafetyError("Could not parse the SQL query.")

    # A query string can contain multiple statements separated by semicolons.
    # We REJECT any input with more than one statement to block `; DROP TABLE`.
    if len(parsed) > 1:
        raise SafetyError(
            "Multiple statements detected. Only a single SELECT is allowed."
        )

    stmt: Statement = parsed[0]

    # Walk through tokens, skip whitespace and comments, find first real keyword
    first_meaningful = None
    for token in stmt.tokens:
        if token.ttype in (sqlparse.tokens.Whitespace,
                           sqlparse.tokens.Newline,
                           sqlparse.tokens.Comment.Single,
                           sqlparse.tokens.Comment.Multiline):
            continue
        first_meaningful = token
        break

    if first_meaningful is None:
        raise SafetyError("Query contains no executable tokens.")

    # The first token of a SELECT must be DML type with value "SELECT".
    # EXCEPTION: CTEs start with "WITH" (sqlparse classifies it as
    # Token.Keyword.CTE, not DML).  A CTE "WITH ... AS (...) SELECT ..."
    # is read-only and should be allowed.
    # We detect CTEs by checking:
    #   - first token value is "WITH" (any Keyword subtype)
    #   - the word SELECT appears somewhere later in the query
    first_val     = first_meaningful.normalized.upper()
    ttype_str     = str(first_meaningful.ttype)   # e.g. "Token.Keyword.CTE"
    is_cte        = (
        first_val == "WITH"
        and "Keyword" in ttype_str
        and re.search(r"\bSELECT\b", query, re.IGNORECASE)
    )

    if not is_cte:
        if first_meaningful.ttype is not DML or first_val != "SELECT":
            raise SafetyError(
                f"Only SELECT statements are permitted. "
                f"Got: '{first_val}'"
            )

    # --- Layer 2: regex blacklist (catches multi-statement injections) ---
    if _BANNED_KEYWORDS.search(query):
        match = _BANNED_KEYWORDS.search(query)
        raise SafetyError(
            f"Banned keyword '{match.group()}' detected in query."
        )


def enforce_limit(query: str, max_rows: int = 500) -> str:
    """
    If the query has no LIMIT clause, append one.
    If it already has a LIMIT that exceeds max_rows, replace it.

    WHY cap rows?
      - Prevents the LLM from accidentally writing `SELECT * FROM sales`
        and flooding the context window with 10 000 rows.
      - Protects against runaway queries on larger databases.
      - Makes response time predictable.
    """
    query = query.rstrip("; \n\t")   # strip trailing semicolons/whitespace

    existing_match = _LIMIT_RE.search(query)
    if existing_match:
        # Extract the existing limit value
        existing_limit_str = existing_match.group()
        existing_limit = int(re.search(r"\d+", existing_limit_str).group())
        if existing_limit > max_rows:
            # Replace with our capped value
            query = _LIMIT_RE.sub(f"LIMIT {max_rows}", query)
    else:
        # No LIMIT present — append one
        query = f"{query} LIMIT {max_rows}"

    return query


def safe_query(query: str, max_rows: int = 500) -> str:
    """
    Full safety pipeline:
      1. validate_sql(query)   — raises SafetyError if not SELECT
      2. enforce_limit(query)  — adds / caps LIMIT clause
      3. returns the sanitised query string

    Usage:
      safe_q = safe_query(user_query)
      conn.execute(safe_q)
    """
    validate_sql(query)             # raises SafetyError on violation
    return enforce_limit(query, max_rows)
