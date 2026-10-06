"""Create and populate the persistent Chroma collection."""

from functools import lru_cache

from langchain_chroma import Chroma

from ..errors import KnowledgeBaseError
from ..ingestion.chunk import load_documents
from .embeddings import get_embeddings
from ..settings import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Open the project's persistent Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def build_vectorstore() -> Chroma:
    documents = load_documents()

    if not documents:
        raise KnowledgeBaseError(
            "Parsed JSON contained no indexable text. Check the PDF parsing output."
        )

    print(f"\nTotal documents: {len(documents)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")

    vectorstore = get_vectorstore()

    existing = vectorstore.get()

    if existing["ids"]:
        print(
            f"Collection already contains "
            f"{len(existing['ids'])} documents."
        )
        print("Skipping insertion.")
        return vectorstore

    print("Adding documents to Chroma...")

    vectorstore.add_documents(documents)

    print(
        f"Saved {len(documents)} documents "
        f"to {CHROMA_DIR}"
    )

    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
