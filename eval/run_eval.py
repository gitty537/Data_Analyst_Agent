"""
eval/run_eval.py
----------------
Runs every question in eval/questions.py through the agent and measures
how often the agent's answer contains the right information.

Scoring strategy (two-tier):
  1. keyword_check  — if `expected_keyword` is set, the answer MUST contain it
  2. number_check   — extract all numbers from both the ground-truth SQL result
                      and the agent's answer; at least one number must match
                      within a 2% tolerance (handles rounding differences)

This is intentionally lightweight — a production eval would use an LLM-as-judge
approach (feeding the agent answer + reference answer to a second LLM), but
that adds latency and cost.  The number-matching approach works well for
analytical Q&A where the answer is always a quantity.

Usage:
  python -m eval.run_eval

Output:
  - Console table with per-question results
  - eval/results.json with full detail
"""

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles to prevent cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path so imports work from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import run_agent
from eval.questions import EVAL_QUESTIONS

DB_PATH     = "data/sales.db"
RESULTS_DIR = "eval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    """
    Extract all numeric values from a string.
    Handles:  1,234.56  |  -45.2  |  0.33  |  1234567
    Returns a deduplicated sorted list of floats.
    """
    # Remove commas inside numbers (e.g. "1,234" → "1234")
    cleaned = text.replace(",", "")
    # Find all floating-point / integer numbers, including negatives
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    return sorted(set(float(n) for n in nums))


def _numbers_match(ref_nums: list[float], agent_nums: list[float],
                   tolerance: float = 0.02) -> bool:
    """
    Return True if any number in ref_nums appears (within tolerance%) in agent_nums.
    Also compares absolute values so 'loss of $33k' matches '-33k'.
    """
    for ref in ref_nums:
        for agent in agent_nums:
            if ref == 0 and agent == 0:
                return True
            if ref != 0:
                # Direct check
                if abs((agent - ref) / ref) <= tolerance:
                    return True
                # Absolute value check (for negative quantities like loss)
                if abs((abs(agent) - abs(ref)) / abs(ref)) <= tolerance:
                    return True
    return False


def run_ground_truth_sql(sql: str) -> str:
    """Execute the ground-truth SQL and return results as a formatted string."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return str(rows)


def score_answer(question: dict, agent_answer: str) -> dict:
    """
    Score the agent's answer against ground truth.

    Returns a dict:
      passed       : bool
      score_method : str   ("keyword" | "number" | "skip")
      details      : str
    """
    # --- Compute ground truth ---
    try:
        gt_result = run_ground_truth_sql(question["ground_truth_sql"])
    except Exception as e:
        return {
            "passed":       False,
            "score_method": "error",
            "details":      f"Ground truth SQL failed: {e}",
        }

    # --- Keyword check ---
    expected_kw = question.get("expected_keyword")
    if expected_kw:
        passed = expected_kw.lower() in agent_answer.lower()
        return {
            "passed":       passed,
            "score_method": "keyword",
            "details":      f"Expected keyword '{expected_kw}' in answer. GT: {gt_result[:200]}",
        }

    # --- Number check ---
    gt_nums    = _extract_numbers(gt_result)
    agent_nums = _extract_numbers(agent_answer)

    if not gt_nums:
        # Ground truth has no numbers (e.g. just a name) — skip numeric check
        return {
            "passed":       True,
            "score_method": "skip",
            "details":      "No numeric ground truth to compare.",
        }

    if not agent_nums:
        return {
            "passed":       False,
            "score_method": "number",
            "details":      f"Agent gave no numbers. GT: {gt_result[:200]}",
        }

    passed = _numbers_match(gt_nums, agent_nums)
    return {
        "passed":       passed,
        "score_method": "number",
        "details": (
            f"GT numbers: {gt_nums[:5]} | Agent numbers: {agent_nums[:5]}"
        ),
    }


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_eval(max_questions: int = None, delay_seconds: float = 1.5):
    """
    Run the full eval suite.

    Parameters
    ----------
    max_questions : int | None
        Limit to first N questions (useful for quick tests).
    delay_seconds : float
        Pause between API calls to respect rate limits.
    """
    questions = EVAL_QUESTIONS
    if max_questions:
        questions = questions[:max_questions]

    print(f"\n{'='*70}")
    print(f"  DATA ANALYST AGENT — EVALUATION SUITE")
    print(f"  {len(questions)} questions")
    print(f"{'='*70}\n")

    results = []
    passed  = 0
    skipped = 0

    for i, q in enumerate(questions, 1):
        print(f"[{i:02d}/{len(questions)}] {q['id']}: {q['question'][:60]}...")

        start = time.time()
        try:
            agent_resp = run_agent(q["question"], verbose=False)
            agent_ans  = agent_resp["answer"]
            elapsed    = time.time() - start
        except Exception as e:
            print(f"  [ERROR] Agent error: {e}")
            results.append({
                "id":           q["id"],
                "question":     q["question"],
                "category":     q["category"],
                "passed":       False,
                "score_method": "error",
                "details":      str(e),
                "agent_answer": "",
                "tool_calls":   0,
                "elapsed_s":    0,
            })
            continue

        # Score the answer
        score = score_answer(q, agent_ans)

        status = "[PASS]" if score["passed"] else "[FAIL]"
        if score["score_method"] == "skip":
            status = "[SKIP]"
            skipped += 1

        print(f"  {status} | {score['score_method']:7s} | {elapsed:.1f}s")
        print(f"  Details: {score['details'][:100]}")

        if score["passed"] or score["score_method"] == "skip":
            if score["score_method"] != "skip":
                passed += 1
        
        results.append({
            "id":           q["id"],
            "question":     q["question"],
            "category":     q["category"],
            "passed":       score["passed"],
            "score_method": score["score_method"],
            "details":      score["details"],
            "agent_answer": agent_ans[:500],
            "tool_calls":   len(agent_resp.get("tool_calls", [])),
            "elapsed_s":    round(elapsed, 2),
        })

        # Rate-limit pause (be a good API citizen)
        if i < len(questions):
            time.sleep(delay_seconds)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_scored = len(results) - skipped
    score_pct    = (passed / total_scored * 100) if total_scored > 0 else 0

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Total questions : {len(results)}")
    print(f"  Scored          : {total_scored}")
    print(f"  Passed          : {passed}")
    print(f"  Failed          : {total_scored - passed}")
    print(f"  Skipped (no GT) : {skipped}")
    print(f"  SCORE           : {score_pct:.1f}%")
    print(f"{'='*70}\n")

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"passed": 0, "total": 0})
        if r["score_method"] != "skip":
            categories[cat]["total"] += 1
            if r["passed"]:
                categories[cat]["passed"] += 1

    print("  Category Breakdown:")
    for cat, counts in sorted(categories.items()):
        total = counts["total"]
        p     = counts["passed"]
        pct   = (p / total * 100) if total > 0 else 0
        print(f"    {cat:12s}: {p}/{total} ({pct:.0f}%)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({
            "score_pct": round(score_pct, 1),
            "passed":    passed,
            "total":     total_scored,
            "results":   results,
        }, f, indent=2)
    print(f"\n  Full results saved to: {out_path}")

    return score_pct


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the agent eval suite.")
    parser.add_argument("--max", type=int, default=None,
                        help="Run only first N questions (for quick tests)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between API calls (default: 1.5)")
    args = parser.parse_args()
    run_eval(max_questions=args.max, delay_seconds=args.delay)
