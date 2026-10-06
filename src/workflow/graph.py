from functools import lru_cache
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from ..errors import LLMServiceError
from ..generation.llm import invoke_llm
from ..generation.rag import build_context, generate_answer

from langchain_core.messages import HumanMessage, SystemMessage
from ..retrieval.retrieval import retrieve


class RAGState(TypedDict):
    question: str
    results: list[Any]
    context: str
    has_context: bool
    answer: str
    error: str | None

def retrieve_node(state: RAGState) -> dict[str, Any]:
    results = retrieve(
        query=state["question"],
        k=3,
    )

    context = build_context(results)

    return {
        "results": results,
        "context": context,
    }


def check_context_node(state: RAGState) -> dict[str, Any]:
    """
    Check whether retrieved context can reasonably answer the question.

    We use the LLM only as a binary relevance judge.
    """

    if not state["context"].strip():
        return {"has_context": False, "error": None}

    system_prompt = """
You are a retrieval relevance evaluator.

Determine whether the provided context contains at least one fact directly
relevant to the user's question. The context does not need to contain every
possible detail, but it must support a useful answer without guessing.

Reply with exactly one word:

YES
or
NO

Do not answer the user's question.
""".strip()

    user_prompt = f"""
Question:
{state["question"]}

Context:
{state["context"]}
""".strip()

    try:
        response = invoke_llm(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            purpose="check whether the retrieved context is relevant",
        )
    except LLMServiceError as exc:
        return {"has_context": False, "error": str(exc)}

    decision = response.content.strip().upper()

    has_context = decision.startswith("YES")

    return {
        "has_context": has_context,
        "error": None,
    }

def answer_node(state: RAGState) -> dict[str, Any]:
    try:
        answer = generate_answer(
            question=state["question"],
            context=state["context"],
        )
    except LLMServiceError as exc:
        return {
            "answer": "An answer cannot be generated right now. Please try again.",
            "error": str(exc),
        }

    return {
        "answer": answer,
        "error": None,
    }


def fallback_node(state: RAGState) -> dict[str, Any]:
    if state.get("error"):
        return {
            "answer": "The question-answering service is unavailable right now. Please try again."
        }

    return {
        "answer": (
            "The available documents do not contain enough information "
            "to answer this question confidently."
        )
    }


def route_context(state: RAGState) -> str:
    if state["has_context"]:
        return "answer"

    return "fallback"


def build_graph() -> CompiledStateGraph:
    """Build the stateless core RAG workflow."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("check_context", check_context_node)
    builder.add_node("answer", answer_node)
    builder.add_node("fallback", fallback_node)
    builder.add_edge(START, "retrieve")

    builder.add_edge("retrieve", "check_context")

    builder.add_conditional_edges(
        "check_context",
        route_context,
        {
            "answer": "answer",
            "fallback": "fallback",
        },
    )

    builder.add_edge("answer", END)
    builder.add_edge("fallback", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """Build and cache the LangGraph workflow. Called lazily on first ask."""
    return build_graph()
