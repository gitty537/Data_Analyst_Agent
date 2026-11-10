"""
agent/tools.py
--------------
The three tools the LLM agent can call:
  1. describe_schema()   — returns table/column metadata as a string
  2. run_sql(query)      — executes a SELECT and returns rows as a list of dicts
  3. make_chart(...)     — saves a PNG chart and returns its file path

WHY expose tools instead of giving the LLM direct DB access?
  - The LLM never touches the database directly — it only calls our Python
    functions.  This is the standard "tool-use" / "function-calling" pattern.
  - Each function validates its inputs, enforces limits, and can log everything.
  - Swapping the backend (SQLite → PostgreSQL) only requires changing these
    three functions, not the agent or the LLM prompt.

Each function is ALSO registered in a JSON schema dict (TOOL_SCHEMAS) that we
pass to the LLM API so it knows how to call them.
"""

import json
import os
import sqlite3
from typing import Any

import matplotlib
# Set writable cache dir before any other matplotlib imports to suppress warnings
import os as _os
_mpl_cache = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".mpl_cache")
_mpl_cache = _os.path.abspath(_mpl_cache)
_os.makedirs(_mpl_cache, exist_ok=True)
_os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

matplotlib.use("Agg")           # non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from agent.safety import safe_query, SafetyError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH    = "data/sales.db"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)   # create charts/ if it doesn't exist


# ===========================================================================
# Tool 1 — describe_schema
# ===========================================================================

def describe_schema() -> str:
    """
    Returns a human-readable description of the database schema.

    WHY this tool?
      The LLM has no prior knowledge of our specific column names.  By calling
      describe_schema() first, it learns the exact names (e.g. "sub_category"
      not "subcategory") and can write correct SQL.

    HOW it works:
      SQLite stores schema info in `sqlite_master`.  PRAGMA table_info() gives
      us column names, types, and nullable info for each table.
    """
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Get all table names from the SQLite metadata table
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cur.fetchall()]

    lines = ["DATABASE SCHEMA", "=" * 40]
    for table in tables:
        lines.append(f"\nTable: {table}")
        lines.append("-" * 30)

        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        cur.execute(f"PRAGMA table_info({table});")
        for cid, name, col_type, notnull, dflt, pk in cur.fetchall():
            pk_marker    = " [PK]"    if pk      else ""
            notnull_mark = " NOT NULL" if notnull else ""
            lines.append(f"  {name:20s} {col_type:10s}{pk_marker}{notnull_mark}")

        # Also show a row count so the LLM knows the data size
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
        lines.append(f"\n  Total rows: {count:,}")

    conn.close()
    return "\n".join(lines)


# ===========================================================================
# Tool 2 — run_sql
# ===========================================================================

def run_sql(query: str) -> dict[str, Any]:
    """
    Executes a SELECT query against the database and returns the results.

    Parameters
    ----------
    query : str
        A SQL SELECT statement.

    Returns
    -------
    dict with keys:
      "columns" : list[str]  — column names
      "rows"    : list[list] — row data
      "row_count": int       — number of rows returned

    WHY return a dict instead of raw rows?
      Dicts are easier for the LLM to interpret when formatted as JSON.
      We always include column names so the LLM can reference them by name.

    Safety pipeline (see agent/safety.py for details):
      1. validate_sql — rejects non-SELECT statements
      2. enforce_limit — caps results at 500 rows
    """
    # ── Safety gate ──────────────────────────────────────────────────────────
    # This is the STRUCTURAL enforcement.  Even if the LLM passes DELETE or
    # DROP, it gets blocked here in Python before touching the DB.
    try:
        # Cap at 100 rows structurally to protect against TPM rate limits
        safe_q = safe_query(query, max_rows=100)
    except SafetyError as e:
        # Return the error as a structured response so the LLM can self-correct
        return {"error": f"Safety violation: {e}", "columns": [], "rows": []}

    # ── Execute ──────────────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row   # makes rows accessible by column name
        cur  = conn.cursor()
        cur.execute(safe_q)
        raw_rows = cur.fetchall()
        columns  = [desc[0] for desc in cur.description] if cur.description else []
        conn.close()
    except sqlite3.Error as e:
        return {"error": f"SQL error: {e}", "columns": [], "rows": []}

    # Convert sqlite3.Row objects to plain lists for JSON serialisability
    # To protect against Free Tier TPM limits (e.g. Groq 8000 TPM cap),
    # return at most 30 sample rows.
    total_found = len(raw_rows)
    display_rows = [list(r) for r in raw_rows[:30]]

    result_payload = {
        "columns": columns,
        "rows": display_rows,
        "row_count": total_found,
    }

    if total_found > 30:
        result_payload["note"] = (
            f"Returned {total_found} rows. Showing first 30 rows to prevent "
            "exceeding LLM token limits. For totals and summaries, use SQL "
            "aggregate functions (SUM, AVG, COUNT, GROUP BY)."
        )

    return result_payload


# ===========================================================================
# Tool 3 — make_chart
# ===========================================================================

def make_chart(
    chart_type: str,
    labels: list[str],
    values: list[float],
    title: str = "Chart",
    x_label: str = "",
    y_label: str = "",
    filename: str = "chart.png",
) -> str:
    """
    Renders a bar or line chart from provided data and saves it as a PNG.

    Parameters
    ----------
    chart_type : "bar" | "line"
    labels     : x-axis categories / time labels
    values     : y-axis numeric values (must be same length as labels)
    title      : chart title
    x_label    : x-axis label
    y_label    : y-axis label
    filename   : output filename (saved inside charts/)

    Returns
    -------
    str — absolute file path of the saved PNG

    WHY matplotlib?
      Lightweight, deterministic output, no browser required.  The Streamlit
      app can display the PNG directly with st.image().

    WHY return a file path instead of the image bytes?
      - File paths are JSON-serialisable (simple string).
      - Streamlit's st.image() accepts a path.
      - Avoids bloating the LLM's context with binary data.
    """
    if len(labels) != len(values):
        raise ValueError(
            f"labels ({len(labels)}) and values ({len(values)}) must have the same length."
        )

    # Defensive casting: LLM sometimes passes numbers as strings from JSON results.
    # This is Failure #3 from the README — coerce to float here to prevent TypeError.
    values = [float(v) for v in values]
    labels = [str(lb) for lb in labels]

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0E1117")  # dark background — matches Streamlit dark mode
    ax.set_facecolor("#1A1D27")

    colors = ["#4C9BE8", "#E86C4C", "#4CE87A", "#E8D34C", "#C04CE8",
              "#4CE8D3", "#E84C8F", "#8FE84C"]

    if chart_type == "bar":
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="#2A2D3A", linewidth=0.8)
        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:,.0f}",
                ha="center", va="bottom", color="white", fontsize=9,
            )

    elif chart_type == "line":
        ax.plot(labels, values, color=colors[0], linewidth=2.5,
                marker="o", markersize=6, markerfacecolor=colors[1])
        ax.fill_between(range(len(labels)), values, alpha=0.15, color=colors[0])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

    else:
        raise ValueError(f"Unsupported chart_type '{chart_type}'. Use 'bar' or 'line'.")

    # ── Styling ──────────────────────────────────────────────────────────────
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(x_label, color="#AAAAAA", fontsize=11)
    ax.set_ylabel(y_label, color="#AAAAAA", fontsize=11)
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2D3A")

    # Format y-axis numbers with commas (e.g. 1,234,567)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Rotate x-labels if there are many (avoids overlap)
    if len(labels) > 6:
        plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(CHARTS_DIR, filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)   # free memory — important in long-running Streamlit sessions

    return os.path.abspath(out_path)


# ===========================================================================
# Tool schemas for the LLM API (OpenAI-compatible function-calling format)
# ===========================================================================
# These JSON objects tell the LLM:
#   - The function's name
#   - What it does (description)
#   - What arguments to pass (parameters)
#
# The LLM uses this metadata to decide WHICH tool to call and WHAT to pass.
# We keep these in sync with the Python function signatures above.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "describe_schema",
            "description": (
                "Returns the full database schema including table names, "
                "column names, data types, and row counts. "
                "Call this FIRST before writing any SQL."
            ),
            "parameters": {
                "type": "object",
                "properties": {},          # no arguments needed
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Executes a SQL SELECT query against the sales database. "
                "Only SELECT is allowed — write, delete, and DDL statements "
                "are blocked. Results are capped at 500 rows. "
                "Returns columns, rows, and row_count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A valid SQLite SELECT statement.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": (
                "Renders a bar or line chart and saves it as a PNG file. "
                "Returns the file path of the saved chart. "
                "Use after run_sql when a visual is helpful."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line"],
                        "description": "Type of chart to render.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "X-axis labels (e.g. region names or dates).",
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Y-axis values, same length as labels.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title.",
                    },
                    "x_label": {
                        "type": "string",
                        "description": "X-axis label.",
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Y-axis label.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output PNG filename (e.g. 'region_sales.png').",
                    },
                },
                "required": ["chart_type", "labels", "values", "title"],
            },
        },
    },
]
