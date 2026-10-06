"""Query the persistent Chroma collection."""

from langchain_core.documents import Document

from .vectorstore import get_vectorstore
from ..errors import KnowledgeBaseError
from ..settings import RETRIEVAL_MAX_DISTANCE


def retrieve(
    query: str,
    k: int = 3,
    max_distance: float | None = RETRIEVAL_MAX_DISTANCE,
    source: str | None = None,
) -> list[tuple[Document, float]]:
    """Return top-k results, rejecting weak matches above max_distance."""
    try:
        vectorstore = get_vectorstore()
        existing = vectorstore.get(limit=1)
    except Exception as exc:
        raise KnowledgeBaseError(
            "Unable to open the Chroma knowledge base. Run 'python -m src.main ingest'."
        ) from exc

    if not existing.get("ids"):
        raise KnowledgeBaseError(
            "The Chroma knowledge base is empty. Run 'python -m src.main ingest' first."
        )

    try:
        search_kwargs = {"k": k}
        if source:
            search_kwargs["filter"] = {"source": source}

        results = vectorstore.similarity_search_with_score(query, **search_kwargs)
        if max_distance is None:
            return results

        return [
            (doc, score)
            for doc, score in results
            if score <= max_distance
        ]
    except Exception as exc:
        raise KnowledgeBaseError("Unable to search the Chroma knowledge base.") from exc
