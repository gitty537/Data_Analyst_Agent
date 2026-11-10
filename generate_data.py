"""
generate_data.py
----------------
Creates a realistic synthetic "Superstore Sales" CSV dataset and saves it to
data/sales.csv.  We build it from scratch so the project is 100% self-contained
(no Kaggle account needed).

WHY synthetic data instead of a live download?
  - The project must run offline / in any CI environment.
  - We can engineer the data so the eval questions have exact, known answers.
  - It still mirrors the real Superstore schema used in Kaggle datasets.

Dataset schema:
  order_id, order_date, ship_date, customer_id, customer_name,
  segment, region, state, city, category, sub_category,
  product_name, sales, quantity, discount, profit
"""

import csv
import random
import uuid
from datetime import date, timedelta

# ── Seed for reproducibility ─────────────────────────────────────────────────
# Using a fixed seed means every regeneration produces the SAME data, so our
# eval answers stay correct.
random.seed(42)

# ── Reference tables ─────────────────────────────────────────────────────────
# These match the real Superstore dataset's categories/regions closely.

REGIONS_STATES = {
    "West":    ["California", "Washington", "Oregon", "Nevada", "Arizona"],
    "East":    ["New York", "Pennsylvania", "Ohio", "Georgia", "Florida"],
    "Central": ["Texas", "Illinois", "Michigan", "Wisconsin", "Missouri"],
    "South":   ["North Carolina", "Virginia", "Tennessee", "Alabama", "Louisiana"],
}

STATE_CITIES = {
    "California":     ["Los Angeles", "San Francisco", "San Diego"],
    "Washington":     ["Seattle", "Tacoma", "Spokane"],
    "Oregon":         ["Portland", "Eugene", "Salem"],
    "Nevada":         ["Las Vegas", "Reno", "Henderson"],
    "Arizona":        ["Phoenix", "Tucson", "Mesa"],
    "New York":       ["New York City", "Buffalo", "Rochester"],
    "Pennsylvania":   ["Philadelphia", "Pittsburgh", "Allentown"],
    "Ohio":           ["Columbus", "Cleveland", "Cincinnati"],
    "Georgia":        ["Atlanta", "Savannah", "Augusta"],
    "Florida":        ["Miami", "Orlando", "Tampa"],
    "Texas":          ["Houston", "Dallas", "Austin"],
    "Illinois":       ["Chicago", "Aurora", "Rockford"],
    "Michigan":       ["Detroit", "Grand Rapids", "Warren"],
    "Wisconsin":      ["Milwaukee", "Madison", "Green Bay"],
    "Missouri":       ["Kansas City", "St. Louis", "Springfield"],
    "North Carolina": ["Charlotte", "Raleigh", "Greensboro"],
    "Virginia":       ["Virginia Beach", "Richmond", "Norfolk"],
    "Tennessee":      ["Nashville", "Memphis", "Knoxville"],
    "Alabama":        ["Birmingham", "Montgomery", "Huntsville"],
    "Louisiana":      ["New Orleans", "Baton Rouge", "Shreveport"],
}

SEGMENTS      = ["Consumer", "Corporate", "Home Office"]
CATEGORIES    = ["Technology", "Furniture", "Office Supplies"]
PRODUCTS      = {
    "Technology": {
        "Phones":      ["iPhone 14", "Samsung Galaxy S22", "Google Pixel 7"],
        "Computers":   ["Dell XPS 15", "MacBook Pro", "Lenovo ThinkPad X1"],
        "Accessories": ["Logitech Keyboard", "Anker USB Hub", "Belkin Charger"],
    },
    "Furniture": {
        "Chairs":      ["Herman Miller Aeron", "Steelcase Leap", "HON Ignition"],
        "Tables":      ["Ikea Linnmon", "Sauder Shoal Creek", "Realspace Magellan"],
        "Bookcases":   ["IKEA BILLY", "Sauder 5-Shelf Bookcase", "Bush Furniture"],
    },
    "Office Supplies": {
        "Binders":     ["Avery Durable", "Cardinal Economy", "Staples Better"],
        "Paper":       ["Hammermill Copy Paper", "HP Printer Paper", "Staples Copy"],
        "Storage":     ["Sterilite 6-Drawer", "IRIS Stack & Pull", "Rubbermaid Brute"],
    },
}

SHIP_MODES    = ["Standard Class", "Second Class", "First Class", "Same Day"]
FIRST_NAMES   = ["James","Mary","John","Patricia","Robert","Jennifer","Michael",
                  "Linda","William","Barbara","David","Susan","Richard","Jessica"]
LAST_NAMES    = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
                 "Davis","Wilson","Taylor","Anderson","Thomas","Jackson","White"]


def random_date(start: date, end: date) -> date:
    """Return a random date between start and end (inclusive)."""
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def build_row(order_date: date, region: str) -> dict:
    """
    Build one row of the sales table.
    All financial figures are computed programmatically so we know the ground
    truth (important for the eval set).
    """
    state     = random.choice(REGIONS_STATES[region])
    city      = random.choice(STATE_CITIES[state])
    segment   = random.choice(SEGMENTS)
    category  = random.choice(CATEGORIES)
    sub_cat   = random.choice(list(PRODUCTS[category].keys()))
    product   = random.choice(PRODUCTS[category][sub_cat])

    # Base unit price varies by category to make data realistic
    base_price = {
        "Technology":      random.uniform(50, 1200),
        "Furniture":       random.uniform(80, 800),
        "Office Supplies": random.uniform(5, 150),
    }[category]

    quantity   = random.randint(1, 10)
    discount   = random.choice([0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    unit_price = round(base_price, 2)
    sales      = round(unit_price * quantity * (1 - discount), 2)

    # Profit margin: Technology is high-margin, Furniture is thin
    margin_range = {
        "Technology":      (0.10, 0.35),
        "Furniture":       (-0.05, 0.20),   # furniture can lose money
        "Office Supplies": (0.05, 0.40),
    }[category]
    margin = random.uniform(*margin_range)
    profit = round(sales * margin, 2)

    ship_days = {"Standard Class": 5, "Second Class": 3,
                 "First Class": 2, "Same Day": 0}
    mode      = random.choice(SHIP_MODES)
    ship_date = order_date + timedelta(days=ship_days[mode] + random.randint(0, 2))

    fname = random.choice(FIRST_NAMES)
    lname = random.choice(LAST_NAMES)

    return {
        "order_id":      f"CA-{order_date.year}-{random.randint(100000,999999)}",
        "order_date":    order_date.isoformat(),
        "ship_date":     ship_date.isoformat(),
        "customer_id":   f"CG-{random.randint(10000,99999)}",
        "customer_name": f"{fname} {lname}",
        "segment":       segment,
        "region":        region,
        "state":         state,
        "city":          city,
        "category":      category,
        "sub_category":  sub_cat,
        "product_name":  product,
        "sales":         sales,
        "quantity":      quantity,
        "discount":      discount,
        "profit":        profit,
    }


def generate_dataset(n_rows: int = 10_000, output_path: str = "data/sales.csv"):
    """
    Generate n_rows rows covering 2021-01-01 to 2024-12-31.

    WHY 10 000 rows?
      Large enough to have statistically meaningful aggregates per region/quarter
      but small enough that SQLite queries run in milliseconds — good for demos.
    """
    start_date = date(2021, 1, 1)
    end_date   = date(2024, 12, 31)

    # We deliberately skew Q4-2023 West region sales DOWN to create a clear
    # "biggest drop" answer for the canonical eval question.  Real projects do
    # the same thing: engineer the dataset so eval answers are deterministic.
    rows = []
    for _ in range(n_rows):
        order_date = random_date(start_date, end_date)
        # Assign region with roughly equal probability
        region = random.choice(list(REGIONS_STATES.keys()))

        row = build_row(order_date, region)

        # --- Inject the "West Q4 2023 drop" signal ---
        # If this row falls in West, Q4 2023, reduce its sales by 60%
        if (row["region"] == "West"
                and order_date >= date(2023, 10, 1)
                and order_date <= date(2023, 12, 31)):
            row["sales"]  = round(row["sales"]  * 0.40, 2)
            row["profit"] = round(row["profit"] * 0.40, 2)

        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Generated {n_rows} rows -> {output_path}")


if __name__ == "__main__":
    generate_dataset()
