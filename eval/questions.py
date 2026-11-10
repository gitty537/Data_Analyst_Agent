"""
eval/questions.py
-----------------
18 evaluation questions with SQL-computable "ground truth" answers.

HOW this eval works:
  1. For each question, we store a SQL query that computes the CORRECT answer
     from the database (ground_truth_sql).
  2. run_eval.py runs the agent on each question, then compares the agent's
     answer to the SQL-computed ground truth.
  3. We use fuzzy matching (not exact string equality) because the LLM may
     phrase "West" as "the West region" — both are correct.

WHY 18 questions?
  Covers a broad range of SQL patterns:
    - Simple aggregation (SUM, AVG, COUNT)
    - Grouping + sorting (GROUP BY, ORDER BY)
    - Time filtering (strftime for quarters/years)
    - Multi-column conditions (WHERE + AND)
    - Derived metrics (profit margin = profit/sales)
    - Ranking (LIMIT 1 for top/bottom)
    - Comparison (q-on-q change)

Question categories:
  A. Region questions    (what the JD specifically asks)
  B. Time questions      (quarters, years)
  C. Product questions   (category, sub-category)
  D. Customer questions  (segments)
  E. Edge cases          (zero discount, negative profit)
"""

EVAL_QUESTIONS = [
    # ── A. Region ────────────────────────────────────────────────────────────
    {
        "id": "A1",
        "question": "Which region had the biggest drop in sales from Q3 2023 to Q4 2023?",
        "ground_truth_sql": """
            WITH q3 AS (
                SELECT region, SUM(sales) AS q3_sales
                FROM sales
                WHERE strftime('%Y', order_date) = '2023'
                  AND CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 7 AND 9
                GROUP BY region
            ),
            q4 AS (
                SELECT region, SUM(sales) AS q4_sales
                FROM sales
                WHERE strftime('%Y', order_date) = '2023'
                  AND CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 10 AND 12
                GROUP BY region
            )
            SELECT q3.region, ROUND(q3.q3_sales - q4.q4_sales, 2) AS drop_amount
            FROM q3 JOIN q4 ON q3.region = q4.region
            ORDER BY drop_amount DESC
            LIMIT 1
        """,
        "expected_keyword": "West",   # the region we engineered to drop
        "category": "region",
    },
    {
        "id": "A2",
        "question": "What is the total sales revenue for each region in 2023?",
        "ground_truth_sql": """
            SELECT region, ROUND(SUM(sales), 2) AS total_sales
            FROM sales
            WHERE strftime('%Y', order_date) = '2023'
            GROUP BY region
            ORDER BY total_sales DESC
        """,
        "expected_keyword": None,     # we just check the agent produces numbers
        "category": "region",
    },
    {
        "id": "A3",
        "question": "Which region has the highest average order value?",
        "ground_truth_sql": """
            SELECT region, ROUND(AVG(sales), 2) AS avg_order_value
            FROM sales
            GROUP BY region
            ORDER BY avg_order_value DESC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "region",
    },
    {
        "id": "A4",
        "question": "How many orders were placed in the West region in 2023?",
        "ground_truth_sql": """
            SELECT COUNT(*) AS order_count
            FROM sales
            WHERE region = 'West'
              AND strftime('%Y', order_date) = '2023'
        """,
        "expected_keyword": None,
        "category": "region",
    },
    {
        "id": "A5",
        "question": "Which region had the highest profit margin in 2022?",
        "ground_truth_sql": """
            SELECT region,
                   ROUND(SUM(profit) / SUM(sales) * 100, 2) AS margin_pct
            FROM sales
            WHERE strftime('%Y', order_date) = '2022'
              AND sales > 0
            GROUP BY region
            ORDER BY margin_pct DESC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "region",
    },

    # ── B. Time / Quarter ────────────────────────────────────────────────────
    {
        "id": "B1",
        "question": "What were total sales in Q4 2023 across all regions?",
        "ground_truth_sql": """
            SELECT ROUND(SUM(sales), 2) AS q4_2023_total
            FROM sales
            WHERE strftime('%Y', order_date) = '2023'
              AND CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 10 AND 12
        """,
        "expected_keyword": None,
        "category": "time",
    },
    {
        "id": "B2",
        "question": "Compare Q1 2024 sales vs Q1 2023 sales. Which year was higher?",
        "ground_truth_sql": """
            SELECT
                strftime('%Y', order_date) AS year,
                ROUND(SUM(sales), 2) AS q1_sales
            FROM sales
            WHERE CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 1 AND 3
              AND strftime('%Y', order_date) IN ('2023', '2024')
            GROUP BY year
            ORDER BY year
        """,
        "expected_keyword": None,
        "category": "time",
    },
    {
        "id": "B3",
        "question": "Which quarter of 2022 had the highest total sales?",
        "ground_truth_sql": """
            SELECT
                CASE
                    WHEN CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 1 AND 3  THEN 'Q1'
                    WHEN CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 4 AND 6  THEN 'Q2'
                    WHEN CAST(strftime('%m', order_date) AS INTEGER) BETWEEN 7 AND 9  THEN 'Q3'
                    ELSE 'Q4'
                END AS quarter,
                ROUND(SUM(sales), 2) AS total_sales
            FROM sales
            WHERE strftime('%Y', order_date) = '2022'
            GROUP BY quarter
            ORDER BY total_sales DESC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "time",
    },
    {
        "id": "B4",
        "question": "What is the month with the highest average daily sales in 2023?",
        "ground_truth_sql": """
            SELECT
                strftime('%m', order_date) AS month,
                ROUND(SUM(sales) / COUNT(DISTINCT order_date), 2) AS avg_daily_sales
            FROM sales
            WHERE strftime('%Y', order_date) = '2023'
            GROUP BY month
            ORDER BY avg_daily_sales DESC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "time",
    },

    # ── C. Product ───────────────────────────────────────────────────────────
    {
        "id": "C1",
        "question": "Which product category generates the most profit?",
        "ground_truth_sql": """
            SELECT category, ROUND(SUM(profit), 2) AS total_profit
            FROM sales
            GROUP BY category
            ORDER BY total_profit DESC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "product",
    },
    {
        "id": "C2",
        "question": "What is the best-selling sub-category by revenue?",
        "ground_truth_sql": """
            SELECT sub_category, ROUND(SUM(sales), 2) AS total_sales
            FROM sales
            GROUP BY sub_category
            ORDER BY total_sales DESC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "product",
    },
    {
        "id": "C3",
        "question": "Which product sub-category has the worst profit margin?",
        "ground_truth_sql": """
            SELECT sub_category,
                   ROUND(SUM(profit) / SUM(sales) * 100, 2) AS margin_pct
            FROM sales
            WHERE sales > 0
            GROUP BY sub_category
            ORDER BY margin_pct ASC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "product",
    },
    {
        "id": "C4",
        "question": "How many unique products are sold in the Technology category?",
        "ground_truth_sql": """
            SELECT COUNT(DISTINCT product_name) AS unique_products
            FROM sales
            WHERE category = 'Technology'
        """,
        "expected_keyword": None,
        "category": "product",
    },

    # ── D. Customer / Segment ────────────────────────────────────────────────
    {
        "id": "D1",
        "question": "Which customer segment has the highest total revenue?",
        "ground_truth_sql": """
            SELECT segment, ROUND(SUM(sales), 2) AS total_revenue
            FROM sales
            GROUP BY segment
            ORDER BY total_revenue DESC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "customer",
    },
    {
        "id": "D2",
        "question": "What percentage of total sales come from the Corporate segment?",
        "ground_truth_sql": """
            SELECT
                ROUND(
                    100.0 * SUM(CASE WHEN segment = 'Corporate' THEN sales ELSE 0 END)
                    / SUM(sales), 2
                ) AS corporate_pct
            FROM sales
        """,
        "expected_keyword": None,
        "category": "customer",
    },
    {
        "id": "D3",
        "question": "Which segment places the most orders?",
        "ground_truth_sql": """
            SELECT segment, COUNT(*) AS order_count
            FROM sales
            GROUP BY segment
            ORDER BY order_count DESC
            LIMIT 1
        """,
        "expected_keyword": None,
        "category": "customer",
    },

    # ── E. Edge cases ────────────────────────────────────────────────────────
    {
        "id": "E1",
        "question": "How many orders have a discount greater than 0?",
        "ground_truth_sql": """
            SELECT COUNT(*) AS discounted_orders
            FROM sales
            WHERE discount > 0
        """,
        "expected_keyword": None,
        "category": "edge",
    },
    {
        "id": "E2",
        "question": "What is the total loss (negative profit) in the Furniture category?",
        "ground_truth_sql": """
            SELECT ROUND(SUM(profit), 2) AS total_loss
            FROM sales
            WHERE category = 'Furniture' AND profit < 0
        """,
        "expected_keyword": None,
        "category": "edge",
    },
    {
        "id": "E3",
        "question": "What is the average discount rate across all orders?",
        "ground_truth_sql": """
            SELECT ROUND(AVG(discount) * 100, 2) AS avg_discount_pct
            FROM sales
        """,
        "expected_keyword": None,
        "category": "edge",
    },
]
