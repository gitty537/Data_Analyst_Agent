"""
quickstart.py
-------------
Run the agent in terminal mode (no Streamlit needed).
Good for testing your API key and seeing the agent work before opening the UI.

Usage:
  python quickstart.py
  python quickstart.py --question "Which region had the biggest drop in sales last quarter?"
"""

import argparse
import json
import os
import sys

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

# Ensure root dir is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent import _resolve_llm_settings, _load_env_file

DEFAULT_QUESTION = "Which region had the biggest drop in sales from Q3 2023 to Q4 2023?"


def test_connection():
    """Verify API credentials and list available models from the provider."""
    from openai import OpenAI

    api_key, base_url, model = _resolve_llm_settings()
    if not api_key:
        print("[Error] No API key found. Set OPENAI_API_KEY or GROQ_API_KEY.")
        sys.exit(1)

    provider = "Groq" if "groq.com" in base_url else "OpenAI"
    print(f"\nTesting connection to {provider}...")
    print(f"  Endpoint : {base_url}")
    print(f"  Key      : {api_key[:6]}...{api_key[-4:]}")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"\n[OK] Connection successful! Found {len(model_ids)} models available to your key:")
        for m in sorted(model_ids):
            marker = " (Default)" if m == model else ""
            print(f"  - {m}{marker}")
    except Exception as e:
        print(f"\n[Error] Connection failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Data Analyst Agent - terminal mode")
    parser.add_argument(
        "--question", "-q",
        default=DEFAULT_QUESTION,
        help="Question to ask the agent",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Override the LLM model (e.g. openai/gpt-oss-20b, gpt-4o-mini)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show each tool call as it happens",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test connection and list accessible models from your provider",
    )
    args = parser.parse_args()

    _load_env_file()

    if args.test:
        test_connection()
        return

    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    api_key, base_url, model = _resolve_llm_settings()

    if not api_key:
        print("\nERROR: No API key set.")
        print("Set it in PowerShell:")
        print("  For Groq (free):   $env:OPENAI_API_KEY = 'gsk_...'")
        print("  For OpenAI:        $env:OPENAI_API_KEY = 'sk-...'")
        print("Or create a .env file in the project folder with: OPENAI_API_KEY=...")
        sys.exit(1)

    provider = "Groq" if "groq.com" in base_url else "OpenAI"
    print(f"\n[Config] Provider: {provider} | Model: {model} | Endpoint: {base_url}")
    print(f"Question: {args.question}")
    print("-" * 60)

    from agent.agent import run_agent
    result = run_agent(args.question, verbose=args.verbose)

    print("\n--- ANSWER ---")
    print(result["answer"])

    if result.get("chart_path"):
        print(f"\nChart saved to: {result['chart_path']}")

    print(f"\n[{result['iterations']} iterations, {len(result['tool_calls'])} tool calls]")


if __name__ == "__main__":
    main()
