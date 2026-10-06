"""Embedding model construction shared by indexing and retrieval."""

import os

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from ..settings import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Create normalized multilingual embeddings for the policy corpus."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False,
    )
