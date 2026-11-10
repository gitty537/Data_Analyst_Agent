"""
Runtime regressions for the agent loop.

The failure this covers: the first model response asks for a tool, and the
next request used to include an assistant message with content set to null.
Groq rejects that payload, so every real question died after the first tool call.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.agent import _assistant_message_dict, _resolve_llm_settings, run_agent


def _tool_message() -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(name="describe_schema", arguments="{}"),
            )
        ],
    )


class _Completions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            message = _tool_message()
        else:
            message = ChatCompletionMessage(role="assistant", content="West is lowest.")
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()


class _FakeOpenAI:
    completions = None

    def __init__(self, *args, **kwargs):
        self.chat = type("Chat", (), {"completions": _Completions()})()
        _FakeOpenAI.completions = self.chat.completions


class TestAssistantMessage(unittest.TestCase):
    def test_tool_call_content_is_a_string(self):
        payload = _assistant_message_dict(_tool_message())
        encoded = json.dumps(payload)

        self.assertEqual(payload["content"], "")
        self.assertNotIn("null", encoded)
        self.assertEqual(payload["tool_calls"][0]["function"]["name"], "describe_schema")
        self.assertNotIn("refusal", payload)
        self.assertNotIn("annotations", payload)

    def test_final_answer_text_is_preserved(self):
        payload = _assistant_message_dict(
            ChatCompletionMessage(role="assistant", content="West")
        )
        self.assertEqual(payload["content"], "West")
        self.assertNotIn("tool_calls", payload)


class TestClientSettings(unittest.TestCase):
    def setUp(self):
        self._saved = {
            name: os.environ.get(name)
            for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "GROQ_API_KEY")
        }
        for name in self._saved:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_openai_key_stays_on_openai(self):
        os.environ["OPENAI_API_KEY"] = "sk-test"
        _key, base_url, model = _resolve_llm_settings()
        self.assertEqual(base_url, "https://api.openai.com/v1")
        self.assertEqual(model, "gpt-4o-mini")

    def test_explicit_base_url_is_not_replaced(self):
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["OPENAI_MODEL"] = "llama-3.3-70b-versatile"
        os.environ["OPENAI_BASE_URL"] = "https://example.test/v1"
        _key, base_url, model = _resolve_llm_settings()
        self.assertEqual(base_url, "https://example.test/v1")
        self.assertEqual(model, "llama-3.3-70b-versatile")

    def test_llama_without_base_url_uses_groq(self):
        os.environ["OPENAI_API_KEY"] = "gsk-test"
        os.environ["OPENAI_MODEL"] = "llama-3.3-70b-versatile"
        _key, base_url, _model = _resolve_llm_settings()
        self.assertEqual(base_url, "https://api.groq.com/openai/v1")


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self._saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-test"

    def tearDown(self):
        if self._saved_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_key

    def test_second_request_has_string_tool_content(self):
        import agent.agent as agent_module

        original = agent_module.OpenAI
        agent_module.OpenAI = _FakeOpenAI
        try:
            result = run_agent("Which region dropped?", verbose=False)
        finally:
            agent_module.OpenAI = original

        calls = _FakeOpenAI.completions.calls
        self.assertGreaterEqual(len(calls), 2)
        assistant = calls[1]["messages"][2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "")
        self.assertIn("tool_calls", assistant)
        self.assertIsInstance(assistant["content"], str)
        self.assertNotIn("null", json.dumps(assistant))
        self.assertEqual(result["answer"], "West is lowest.")
        self.assertGreaterEqual(len(result["tool_calls"]), 1)


if __name__ == "__main__":
    unittest.main()
