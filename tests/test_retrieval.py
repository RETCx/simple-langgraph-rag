"""Tests for retrieval logic (src/retrieval/retrieval.py)."""

import unittest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.errors import KnowledgeBaseError
from src.retrieval import retrieval


class _PopulatedVectorstore:
    """Mock vectorstore that returns predefined results."""

    def __init__(self, results):
        self._results = results

    def get(self, limit=None):
        return {"ids": ["id-1"]}

    def similarity_search_with_score(self, query, k=3, filter=None):
        self.last_filter = filter
        return self._results[:k]


class _EmptyVectorstore:
    def get(self, limit=None):
        return {"ids": []}


class TestRetrieveErrors(unittest.TestCase):
    def test_empty_knowledge_base_raises(self):
        with patch.object(retrieval, "get_vectorstore", return_value=_EmptyVectorstore()):
            with self.assertRaises(KnowledgeBaseError):
                retrieval.retrieve("any question")

    def test_broken_vectorstore_raises(self):
        with patch.object(retrieval, "get_vectorstore", side_effect=RuntimeError("corrupt")):
            with self.assertRaises(KnowledgeBaseError):
                retrieval.retrieve("any question")


class TestRetrieveResults(unittest.TestCase):
    def test_returns_expected_documents(self):
        doc = Document(page_content="test content", metadata={"source": "test"})
        mock_results = [(doc, 0.85)]
        vs = _PopulatedVectorstore(mock_results)

        with patch.object(retrieval, "get_vectorstore", return_value=vs):
            results = retrieval.retrieve("test query", k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].page_content, "test content")
        self.assertAlmostEqual(results[0][1], 0.85)

    def test_k_limits_results(self):
        docs = [
            (Document(page_content=f"doc {i}", metadata={}), 0.9 - i * 0.1)
            for i in range(5)
        ]
        vs = _PopulatedVectorstore(docs)

        with patch.object(retrieval, "get_vectorstore", return_value=vs):
            results = retrieval.retrieve("test", k=2, max_distance=None)

        self.assertEqual(len(results), 2)

    def test_filters_results_above_distance_threshold(self):
        docs = [
            (Document(page_content="relevant", metadata={}), 0.50),
            (Document(page_content="weak", metadata={}), 0.91),
        ]
        vs = _PopulatedVectorstore(docs)

        with patch.object(retrieval, "get_vectorstore", return_value=vs):
            results = retrieval.retrieve("test", k=2, max_distance=0.85)

        self.assertEqual([doc.page_content for doc, _score in results], ["relevant"])

    def test_applies_exact_source_filter(self):
        doc = Document(page_content="saving plan", metadata={})
        vs = _PopulatedVectorstore([(doc, 0.50)])

        with patch.object(retrieval, "get_vectorstore", return_value=vs):
            retrieval.retrieve(
                "saving",
                source="document-a",
            )

        self.assertEqual(
            vs.last_filter,
            {"source": "document-a"},
        )


if __name__ == "__main__":
    unittest.main()
