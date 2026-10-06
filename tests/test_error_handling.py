"""Focused tests for expected, user-facing failure paths."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.errors import (
    InputDataError,
    KnowledgeBaseError,
    LLMServiceError,
    ParsedDataError,
)
from src.generation import llm
from src.ingestion import chunk, ingest
from src.retrieval import retrieval


class _EmptyVectorstore:
    def get(self, limit):
        return {"ids": []}


class _BrokenLLM:
    def invoke(self, messages):
        raise RuntimeError("provider unavailable")


class ErrorHandlingTests(unittest.TestCase):
    def test_missing_pdf_folder_is_explained(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_dir = Path(temporary_dir) / "missing"
            with patch.object(ingest, "PDF_DIR", missing_dir):
                with self.assertRaises(InputDataError):
                    ingest.parse_pdfs()

    def test_missing_parsed_json_is_explained(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_dir = Path(temporary_dir) / "missing"
            with patch.object(chunk, "PARSED_DIR", missing_dir):
                with self.assertRaises(ParsedDataError):
                    chunk.load_documents()

    def test_malformed_json_structure_is_rejected(self):
        with self.assertRaises(ParsedDataError):
            chunk.validate_parsed_document({}, "broken.json")

    def test_empty_chroma_is_explained(self):
        with patch.object(retrieval, "get_vectorstore", return_value=_EmptyVectorstore()):
            with self.assertRaises(KnowledgeBaseError):
                retrieval.retrieve("test question")

    def test_llm_provider_failure_is_explained(self):
        with patch.object(llm, "get_llm", return_value=_BrokenLLM()):
            with self.assertRaises(LLMServiceError):
                llm.invoke_llm([], "generate an answer")


if __name__ == "__main__":
    unittest.main()
