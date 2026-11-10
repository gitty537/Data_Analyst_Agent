"""
app.py
------
Streamlit web UI for the Data Analyst Agent.

WHY Streamlit?
  - Zero boilerplate: a 50-line script becomes a full web app.
  - Chat interface built-in with st.chat_message and st.chat_input.
  - Renders charts, tables, and markdown out of the box.
  - Stateful with st.session_state — perfect for a multi-turn chat history.

Run with:
  streamlit run app.py
"""

import json
import os
import sys

import streamlit as st

# ── Page config must be the FIRST Streamlit call ─────────────────────────────
st.set_page_config(
    page_title = "Data Analyst Agent",
    page_icon  = "📊",
    layout     = "wide",
)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    provider = st.radio(
        "LLM Provider",
        ["Groq (Free)", "OpenAI"],
        index=0,
        help="Groq offers fast, free tier inference. OpenAI uses standard GPT models.",
    )

    if provider == "Groq (Free)":
        groq_models = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
        model = st.selectbox(
            "Model",
            groq_models,
            index=0,
            help="'openai/gpt-oss-20b' is Groq's fast default with native tool-calling.",
        )
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", "") or os.environ.get("OPENAI_API_KEY", ""),
            help="Starts with 'gsk_'. Get one free at console.groq.com/keys",
        )
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
        os.environ["OPENAI_MODEL"] = model

    else:
        openai_models = ["gpt-4o-mini", "gpt-4o"]
        model = st.selectbox(
            "Model",
            openai_models,
            index=0,
            help="'gpt-4o-mini' is cheap and fast; 'gpt-4o' handles complex tasks.",
        )
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", "") if not os.environ.get("OPENAI_API_KEY", "").startswith("gsk_") else "",
            help="Starts with 'sk-'. Get one at platform.openai.com/api-keys",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ["OPENAI_MODEL"] = model

    st.divider()
    st.markdown("### 📂 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Upload your own dataset to query and analyze it instantly.",
    )
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file") != file_id:
            with st.spinner("Ingesting dataset into database..."):
                try:
                    from agent.ingest import ingest_dataset
                    meta = ingest_dataset(uploaded_file, table_name="uploaded_dataset")
                    st.session_state["last_uploaded_file"] = file_id
                    st.session_state["uploaded_meta"] = meta
                    st.success(f"Loaded `{uploaded_file.name}` ({meta['row_count']:,} rows)")
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": (
                            f"📂 **Custom Dataset Activated:** Successfully ingested `{uploaded_file.name}` "
                            f"into table `uploaded_dataset` ({meta['row_count']:,} rows, {len(meta['columns'])} columns: "
                            f"`{', '.join(meta['columns'][:8])}{'...' if len(meta['columns']) > 8 else ''}`).\n\n"
                            "You can now ask questions about this dataset, such as *'Summarize this dataset'*, "
                            "*'Show top 5 records by value'*, or request custom charts!"
                        ),
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")

    if st.session_state.get("uploaded_meta"):
        meta = st.session_state["uploaded_meta"]
        st.caption(f"Active Table: `uploaded_dataset` ({meta['row_count']:,} rows)")

    st.divider()
    st.markdown("### 💡 Example Questions")
    if st.session_state.get("uploaded_meta"):
        example_questions = [
            "Summarize the key metrics in uploaded_dataset",
            "Show the top 5 records by highest values",
            "Which category or group appears most frequently?",
            "Make a bar chart representing the top records",
        ]
    else:
        example_questions = [
            "Which region had the biggest drop in sales last quarter?",
            "What is the best-selling product category?",
            "Show me monthly sales trend for 2023",
            "Which segment has the highest profit margin?",
            "Compare Q4 2023 vs Q3 2023 sales by region",
        ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["prefill_question"] = q

    st.divider()
    st.markdown("### 🛡️ Safety")
    st.info(
        "All SQL is structurally enforced:\n"
        "- Only SELECT allowed\n"
        "- Max 100 rows per query\n"
        "- Multi-statement injection blocked"
    )

    if st.button("🗑️ Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📊 Data Analyst Agent")
st.caption(
    "Ask questions in plain English — the agent writes SQL, runs it, "
    "and optionally draws a chart."
)

# Initialise message history if first run
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render existing conversation
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Re-display charts if present
        if msg.get("chart_path") and os.path.exists(msg["chart_path"]):
            st.image(msg["chart_path"])
        # Re-display tool call expander if present
        if msg.get("tool_calls"):
            with st.expander("🔧 Tool calls", expanded=False):
                st.json(msg["tool_calls"])

# ── Handle prefilled question from sidebar button ─────────────────────────
prefill = st.session_state.pop("prefill_question", None)

# Chat input
user_input = st.chat_input(
    "Ask a question about sales data...",
    key="chat_input",
)

# Use prefilled text if a sidebar button was clicked
question = prefill or user_input

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    # Check API key before calling the agent
    if not os.environ.get("OPENAI_API_KEY"):
        with st.chat_message("assistant"):
            st.error("⚠️ Please enter your OpenAI API key in the sidebar first.")
        st.stop()

    # Run agent with a spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from agent.agent import run_agent
                result = run_agent(question, verbose=False)
            except Exception as e:
                st.error(f"Agent error: {e}")
                st.stop()

        # Display the final answer
        st.markdown(result["answer"])

        # Display chart if one was generated
        chart_path = result.get("chart_path")
        if chart_path and os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)

        # Expandable tool call log (great for debugging / transparency)
        if result.get("tool_calls"):
            with st.expander(
                f"🔧 {len(result['tool_calls'])} tool call(s) — click to inspect",
                expanded=False,
            ):
                for i, tc in enumerate(result["tool_calls"], 1):
                    st.markdown(f"**Step {i}: `{tc['tool']}`**")
                    if tc.get("arguments"):
                        st.code(
                            json.dumps(tc["arguments"], indent=2),
                            language="json",
                        )
                    if tc.get("result"):
                        # For SQL results, show as a pretty table if possible
                        try:
                            res = json.loads(tc["result"])
                            if "rows" in res and "columns" in res:
                                import pandas as pd
                                df = pd.DataFrame(res["rows"], columns=res["columns"])
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.code(tc["result"][:1000], language="json")
                        except Exception:
                            st.code(tc["result"][:1000])
                    st.divider()

        # Metrics row
        col1, col2 = st.columns(2)
        col1.metric("Tool calls", len(result.get("tool_calls", [])))
        col2.metric("Agent iterations", result.get("iterations", 0))

    # Save to conversation history
    st.session_state["messages"].append({
        "role":       "assistant",
        "content":    result["answer"],
        "chart_path": result.get("chart_path"),
        "tool_calls": result.get("tool_calls", []),
    })
