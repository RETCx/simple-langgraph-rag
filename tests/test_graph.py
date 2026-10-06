"""Tests for the LangGraph answer/fallback workflow (src/workflow/graph.py)."""

import unittest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.errors import LLMServiceError
from src.workflow.graph import (
    RAGState,
    answer_node,
    check_context_node,
    fallback_node,
    route_context,
)


class TestRouteContext(unittest.TestCase):
    def test_routes_to_answer_when_context_present(self):
        state: RAGState = {
            "question": "q",
            "results": [],
            "context": "some context",
            "has_context": True,
            "answer": "",
            "error": None,
        }
        self.assertEqual(route_context(state), "answer")

    def test_routes_to_fallback_when_no_context(self):
        state: RAGState = {
            "question": "q",
            "results": [],
            "context": "",
            "has_context": False,
            "answer": "",
            "error": None,
        }
        self.assertEqual(route_context(state), "fallback")


class TestFallbackNode(unittest.TestCase):
    def test_fallback_on_insufficient_context(self):
        state: RAGState = {
            "question": "q",
            "results": [],
            "context": "",
            "has_context": False,
            "answer": "",
            "error": None,
        }
        result = fallback_node(state)
        self.assertIn("do not contain enough information", result["answer"])

    def test_fallback_on_llm_error(self):
        state: RAGState = {
            "question": "q",
            "results": [],
            "context": "",
            "has_context": False,
            "answer": "",
            "error": "LLM provider unavailable",
        }
        result = fallback_node(state)
        self.assertIn("service is unavailable", result["answer"])


class TestCheckContextNode(unittest.TestCase):
    def test_empty_context_is_rejected_without_calling_llm(self):
        state: RAGState = {
            "question": "out of scope",
            "results": [],
            "context": "",
            "has_context": False,
            "answer": "",
            "error": None,
        }

        with patch("src.workflow.graph.invoke_llm") as invoke_mock:
            result = check_context_node(state)

        invoke_mock.assert_not_called()
        self.assertFalse(result["has_context"])

    def test_llm_failure_sets_no_context(self):
        with patch(
            "src.workflow.graph.invoke_llm",
            side_effect=LLMServiceError("provider down"),
        ):
            state: RAGState = {
                "question": "q",
                "results": [],
                "context": "some context",
                "has_context": False,
                "answer": "",
                "error": None,
            }
            result = check_context_node(state)

        self.assertFalse(result["has_context"])
        self.assertIsNotNone(result["error"])

    def test_llm_yes_sets_context_true(self):
        mock_response = MagicMock()
        mock_response.content = "YES"
        with patch("src.workflow.graph.invoke_llm", return_value=mock_response):
            state: RAGState = {
                "question": "q",
                "results": [],
                "context": "relevant document information",
                "has_context": False,
                "answer": "",
                "error": None,
            }
            result = check_context_node(state)

        self.assertTrue(result["has_context"])

    def test_llm_no_sets_context_false(self):
        mock_response = MagicMock()
        mock_response.content = "NO"
        with patch("src.workflow.graph.invoke_llm", return_value=mock_response):
            state: RAGState = {
                "question": "q",
                "results": [],
                "context": "irrelevant text",
                "has_context": False,
                "answer": "",
                "error": None,
            }
            result = check_context_node(state)

        self.assertFalse(result["has_context"])


class TestAnswerNode(unittest.TestCase):
    def test_llm_failure_returns_fallback_message(self):
        with patch(
            "src.workflow.graph.generate_answer",
            side_effect=LLMServiceError("timeout"),
        ):
            state: RAGState = {
                "question": "q",
                "results": [],
                "context": "context",
                "has_context": True,
                "answer": "",
                "error": None,
            }
            result = answer_node(state)

        self.assertIn("cannot be generated", result["answer"])
        self.assertIsNotNone(result["error"])

    def test_successful_answer(self):
        with patch(
            "src.workflow.graph.generate_answer",
            return_value="Notice must be given within 30 days.",
        ):
            state: RAGState = {
                "question": "q",
                "results": [],
                "context": "context",
                "has_context": True,
                "answer": "",
                "error": None,
            }
            result = answer_node(state)

        self.assertEqual(result["answer"], "Notice must be given within 30 days.")
        self.assertIsNone(result["error"])


if __name__ == "__main__":
    unittest.main()
