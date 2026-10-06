"""Command-line entry point for building and querying the document RAG system."""

import argparse
import logging
import os
import warnings

# Configure dependency libraries before importing modules that initialize them.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from .workflow.graph import get_graph
from .errors import RAGApplicationError
from .ingestion.ingest import parse_pdfs
from .retrieval.vectorstore import build_vectorstore


def configure_cli_output():
    """Hide dependency diagnostics while preserving application messages."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "openai",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    warnings.filterwarnings(
        "ignore",
        message=".*unauthenticated requests to the HF Hub.*",
    )

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except ImportError:
        pass


def create_initial_state(question: str = ""):
    return {
        "question": question,
        "results": [],
        "context": "",
        "has_context": False,
        "answer": "",
        "error": None,
    }


def print_sources(results):
    print("\n" + "=" * 100)
    print("SOURCES")
    print("=" * 100)

    for rank, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata

        print(
            f"{rank}. "
            f"{metadata.get('source')} | "
            f"page {metadata.get('page_start')}-"
            f"{metadata.get('page_end')} | "
            f"section={metadata.get('section')} | "
            f"subsection={metadata.get('subsection')} | "
            f"score={score:.4f}"
        )


def print_flow_update(node, update, state):
    """Explain each completed LangGraph stage in user-friendly language."""
    if node == "retrieve":
        count = len(update.get("results", []))
        print(f"[FLOW] 1. Retrieval: ChromaDB returned {count} result(s)")
        print("[FLOW] 2. Context check: checking whether the documents can answer the question")

    elif node == "check_context":
        if update.get("has_context"):
            print("[FLOW] 2. Context check: sufficient information; generating an answer")
        else:
            print("[FLOW] 2. Context check: insufficient information; using the fallback")

    elif node == "answer":
        print("[FLOW] 3. Answer: generated an answer from the documents")

    elif node == "fallback":
        print("[FLOW] 3. Fallback: returned a response without guessing beyond the documents")


def run_graph(state, show_progress=True):
    graph = get_graph()
    current_state = dict(state)

    if show_progress:
        print("\n[FLOW] Starting the LangGraph workflow")

    for event in graph.stream(
        state,
        stream_mode="updates",
    ):
        for node, update in event.items():
            if update:
                current_state.update(update)

            if show_progress:
                print_flow_update(
                    node,
                    update or {},
                    current_state,
                )
    return current_state


def ask(question, show_progress=True):
    result = run_graph(
        create_initial_state(question),
        show_progress=show_progress,
    )

    print("\n" + "=" * 100)
    print("ANSWER")
    print("=" * 100)
    print(result["answer"])

    if result.get("results"):
        print_sources(result["results"])

    return result


def cmd_ingest(_args):
    parse_pdfs()
    build_vectorstore()


def cmd_ask(args):
    ask(
        args.question,
        show_progress=not args.quiet,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Document RAG command line")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser(
        "ingest",
        help="Parse PDFs and build the Chroma knowledge base",
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = commands.add_parser(
        "ask",
        help="Ask one question about the indexed documents",
    )
    ask_parser.add_argument(
        "question",
        help="Question to ask.",
    )
    ask_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide LangGraph progress messages and print only the result",
    )

    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main():
    configure_cli_output()
    args = build_parser().parse_args()
    try:
        args.func(args)
    except RAGApplicationError as exc:
        print(f"\nError: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
