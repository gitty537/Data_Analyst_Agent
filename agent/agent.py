"""
agent/agent.py
--------------
The core agent loop:  user question → LLM → tool calls → LLM → answer.

Pattern used: ReAct-style agentic loop
  Reason (the LLM thinks about what to do)
  Act     (the LLM calls a tool)
  Observe (we return the tool result)
  Repeat until the LLM produces a final text answer (no more tool calls).

WHY a loop instead of a one-shot call?
  Some questions need multiple steps:
    1. describe_schema()           → learn column names
    2. run_sql("SELECT ...")       → get the data
    3. make_chart(...)             → visualise it
  A single LLM call can't do all three; a loop can.

WHY NOT use LangChain / LlamaIndex here?
  For learning purposes, writing the loop from scratch shows you exactly what
  those frameworks do under the hood.  You can always wrap this in LangChain
  later once you understand the primitives.

API compatibility:
  This module targets the OpenAI-compatible chat-completion API.
  Any provider that implements the same interface works:
    - OpenAI (gpt-4o, gpt-4-turbo)
    - Groq   (llama-3.1-70b-versatile)  ← FREE, fast, great for demos
    - Mistral, Together.ai, Anyscale, etc.
"""

import json
import os
import time
from typing import Any

from openai import OpenAI            # pip install openai  (works for all providers)

from agent.tools import TOOL_SCHEMAS, describe_schema, make_chart, run_sql

# ---------------------------------------------------------------------------
# Configuration & Auto-Environment Loading
# ---------------------------------------------------------------------------

def _load_env_file():
    """Load key/value pairs from .env in project root if present and not already in os.environ."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

_load_env_file()

# Max iterations of the tool-call loop (safety net — prevents infinite loops)
MAX_ITERATIONS = 10

# Response-only fields the chat-completions endpoint must not receive back.
_ASSISTANT_DROP_KEYS = ("refusal", "annotations", "audio", "function_call")

# Decommissioned/deprecated models mapped to current active alternatives
DEPRECATED_MODELS = {
    "llama-3.1-70b-versatile": "openai/gpt-oss-20b",
    "llama-3.1-8b-instant":    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile": "openai/gpt-oss-20b",
    "llama3-70b-8192":         "openai/gpt-oss-20b",
    "llama3-8b-8192":          "openai/gpt-oss-20b",
}


def _resolve_llm_settings() -> tuple[str, str, str]:
    """
    Return (api_key, base_url, model) from the environment.

    Auto-detects:
      - Groq API keys (start with 'gsk_'): routes to Groq endpoint and uses active
        tool-calling model 'openai/gpt-oss-20b' by default.
      - OpenAI API keys (start with 'sk-'): routes to OpenAI endpoint and uses
        'gpt-4o-mini' by default.
      - Automatically remaps deprecated model names to prevent 400/404 errors.
    """
    _load_env_file()

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    groq_key   = os.environ.get("GROQ_API_KEY", "").strip()
    api_key    = openai_key or groq_key

    explicit_base = os.environ.get("OPENAI_BASE_URL", "").strip()
    model         = os.environ.get("OPENAI_MODEL", "").strip()

    is_groq_key = api_key.startswith("gsk_") or bool(groq_key and not openai_key)

    if is_groq_key:
        # User is using Groq
        base_url = explicit_base if ("groq.com" in explicit_base) else "https://api.groq.com/openai/v1"
        if not model or model.startswith("gpt-4") or model.startswith("gpt-3"):
            model = "openai/gpt-oss-20b"
        elif model in DEPRECATED_MODELS:
            print(f"[Notice] Groq deprecated '{model}'. Auto-migrated to '{DEPRECATED_MODELS[model]}'.")
            model = DEPRECATED_MODELS[model]
    else:
        # Standard OpenAI or custom proxy
        base_url = explicit_base or "https://api.openai.com/v1"
        if not model or model.startswith("llama") or "gpt-oss" in model:
            model = "gpt-4o-mini"

    return api_key, base_url, model


def _assistant_message_dict(message: Any) -> dict[str, Any]:
    """
    Turn an SDK assistant message into a plain chat message.

    Tool-call turns come back with content=None, plus response-only fields
    such as refusal and annotations. The next request then contains
    `"content": null`. OpenAI accepts that; Groq (the default in the Streamlit
    app) rejects the whole turn with a 400, so the agent dies after the first
    tool call. Compatible servers require content to be a string.
    """
    if hasattr(message, "model_dump"):
        raw = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        raw = {key: value for key, value in message.items() if value is not None}
    else:
        raw = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            raw["tool_calls"] = tool_calls

    for key in _ASSISTANT_DROP_KEYS:
        raw.pop(key, None)

    content = raw.get("content")
    payload: dict[str, Any] = {
        "role": raw.get("role", "assistant"),
        "content": content if isinstance(content, str) else "",
    }

    tool_calls = raw.get("tool_calls") or []
    if tool_calls:
        cleaned = []
        for tool_call in tool_calls:
            if hasattr(tool_call, "model_dump"):
                tool_call = tool_call.model_dump(exclude_none=True)
            function = tool_call.get("function") or {}
            if hasattr(function, "model_dump"):
                function = function.model_dump(exclude_none=True)
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            cleaned.append({
                "id": tool_call.get("id"),
                "type": tool_call.get("type") or "function",
                "function": {
                    "name": function.get("name"),
                    "arguments": arguments,
                },
            })
        payload["tool_calls"] = cleaned

    return payload


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """
    Given a tool name and its parsed arguments (from the LLM's tool call),
    call the corresponding Python function and return its result as a JSON string.

    WHY JSON string?
      The OpenAI API expects tool results to be strings in the message history.
      We serialize with json.dumps so structured data (dicts, lists) survives
      the round-trip cleanly.
    """
    if name == "describe_schema":
        result = describe_schema()

    elif name == "run_sql":
        query  = arguments.get("query", "")
        result = run_sql(query)

    elif name == "make_chart":
        result = make_chart(
            chart_type = arguments.get("chart_type", "bar"),
            labels     = arguments.get("labels",     []),
            values     = arguments.get("values",     []),
            title      = arguments.get("title",      "Chart"),
            x_label    = arguments.get("x_label",    ""),
            y_label    = arguments.get("y_label",    ""),
            filename   = arguments.get("filename",   "chart.png"),
        )

    else:
        result = {"error": f"Unknown tool: {name}"}

    # Always convert to string — the API requires it
    if isinstance(result, str):
        return result
    return json.dumps(result)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data analyst assistant with access to an analytical SQLite database. The database contains a default 'sales' table and may also contain user-uploaded datasets (e.g. 'uploaded_dataset').

Your job:
1. Call describe_schema() FIRST to inspect available tables and their exact column names.
2. Formulate an optimized SQL SELECT query and call run_sql(query) to query the appropriate table.
3. If the user asks for a chart or if a visual comparison adds value, call make_chart(...) to visualize results.
4. Provide a clear, insightful answer summarizing the findings.

Rules:
- Only write SELECT queries — no INSERT, UPDATE, DELETE, DROP, etc.
- Always reference the exact table and column names returned by describe_schema().
- Always write SQL queries that aggregate data (e.g. SUM, AVG, COUNT, GROUP BY) or include a LIMIT clause (e.g. LIMIT 10). Never run unbounded SELECT queries that fetch hundreds of raw rows.
- When comparing time periods (e.g. quarters), use strftime() for SQLite dates.
  Example: strftime('%Y', order_date) = '2023' AND strftime('%m', order_date) BETWEEN '10' AND '12'
- Round monetary values to 2 decimal places in your answer.
- If a query returns an error, explain what went wrong and try again with a corrected query.
"""

# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_agent(
    user_question: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run the full agent loop for a single user question.

    Parameters
    ----------
    user_question : str
        The natural-language question from the user.
    verbose : bool
        If True, print each step (tool calls and results) to stdout.

    Returns
    -------
    dict with keys:
      "answer"      : str            — the LLM's final text answer
      "chart_path"  : str | None     — path to chart PNG if one was made
      "tool_calls"  : list[dict]     — log of every tool called and its result
      "iterations"  : int            — how many loop iterations were needed
    """
    api_key, base_url, model = _resolve_llm_settings()

    if not api_key:
        raise EnvironmentError(
            "API Key is missing.\n"
            "Set it via environment variable or in your Streamlit app sidebar."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    # ── Message history ───────────────────────────────────────────────────────
    # The LLM is stateless — we maintain the full conversation history ourselves
    # and pass it on every API call.  This is the standard chat-completion
    # pattern; each message has a "role" and "content".
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_question},
    ]

    tool_call_log = []     # record of every tool call for debugging / eval
    chart_path    = None   # will be set if make_chart is called
    iterations    = 0

    # ── Agent loop ────────────────────────────────────────────────────────────
    while iterations < MAX_ITERATIONS:
        iterations += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {iterations}")
            print(f"{'='*60}")

        # Call the LLM
        # We pass tools=TOOL_SCHEMAS so the LLM knows what functions exist.
        # tool_choice="auto" means the LLM decides whether to call a tool.
        try:
            response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                tools       = TOOL_SCHEMAS,
                tool_choice = "auto",
                temperature = 0,       # 0 = deterministic — important for eval
            )
        except Exception as e:
            err_str = str(e)
            if "413" in err_str or "too large" in err_str.lower() or "tpm" in err_str.lower():
                # Token limit reached (e.g. Groq free tier 8000 TPM cap)
                # Aggressively compact previous tool call outputs and retry
                print(f"\n[Notice] Request size exceeded provider TPM limit. Compacting message history...")
                for m in messages:
                    if isinstance(m, dict) and m.get("role") == "tool" and isinstance(m.get("content"), str):
                        if len(m["content"]) > 400:
                            m["content"] = m["content"][:400] + "\n... [trimmed to fit token limit]"
                response = client.chat.completions.create(
                    model       = model,
                    messages    = messages,
                    tools       = TOOL_SCHEMAS,
                    tool_choice = "auto",
                    temperature = 0,
                )
            elif ("404" in err_str or "400" in err_str) and ("model" in err_str.lower() or "decommissioned" in err_str.lower()):
                fallback = "openai/gpt-oss-20b" if "groq.com" in base_url else "gpt-4o-mini"
                if model != fallback:
                    print(f"\n[Warning] Model '{model}' unavailable. Retrying with active model '{fallback}'...")
                    model = fallback
                    response = client.chat.completions.create(
                        model       = model,
                        messages    = messages,
                        tools       = TOOL_SCHEMAS,
                        tool_choice = "auto",
                        temperature = 0,
                    )
                else:
                    raise e
            else:
                raise e

        # The response has one "choice".  Its message may contain:
        #   a) text content only → LLM is done, return the answer
        #   b) tool_calls only   → LLM wants to call tool(s)
        #   c) both              → unusual but possible
        choice  = response.choices[0]
        message = choice.message

        # Append a provider-safe dict. The raw SDK object serialises tool-call
        # turns as `"content": null`, which Groq rejects on the next request.
        messages.append(_assistant_message_dict(message))

        # ── Case A: No tool calls → final answer ─────────────────────────────
        if not message.tool_calls:
            answer = message.content or ""
            if verbose:
                try:
                    print(f"\n[FINAL ANSWER]\n{answer}")
                except UnicodeEncodeError:
                    safe_text = answer.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
                    print(f"\n[FINAL ANSWER]\n{safe_text}")
            return {
                "answer":     answer,
                "chart_path": chart_path,
                "tool_calls": tool_call_log,
                "iterations": iterations,
            }

        # ── Case B: Tool calls ────────────────────────────────────────────────
        # The LLM may request multiple tool calls in one turn (parallel calls).
        # We handle them all, then continue the loop.
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name

            # Arguments arrive as a JSON string — parse it
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            if verbose:
                print(f"\n[TOOL CALL] {fn_name}({json.dumps(fn_args, indent=2)})")

            # Dispatch to our Python function
            result_str = _dispatch_tool(fn_name, fn_args)

            if verbose:
                # Only print first 500 chars of result to keep terminal clean
                preview = result_str[:500] + ("..." if len(result_str) > 500 else "")
                print(f"[TOOL RESULT] {preview}")

            # Track chart path if make_chart was called
            if fn_name == "make_chart":
                try:
                    chart_path = result_str.strip('"')   # it's a plain string path
                except Exception:
                    pass

            # Log the tool call
            tool_call_log.append({
                "tool":      fn_name,
                "arguments": fn_args,
                "result":    result_str[:2000],  # cap stored result at 2000 chars
            })

            # Add the tool result to message history.
            # To protect against Free Tier TPM limits (e.g. Groq 8000 TPM limit),
            # cap the payload content stored in the LLM's active message history.
            safe_content = result_str
            if len(safe_content) > 2000:
                safe_content = safe_content[:2000] + "\n... [truncated to prevent rate limit]"

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      safe_content,
            })

    # If we exit the loop without a final answer, something went wrong
    return {
        "answer":     "Agent reached max iterations without a final answer.",
        "chart_path": chart_path,
        "tool_calls": tool_call_log,
        "iterations": iterations,
    }
