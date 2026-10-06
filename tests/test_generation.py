import unittest

from langchain_core.documents import Document

from src.generation.rag import build_context


class TestContextBuilder(unittest.TestCase):
    def test_includes_document_content_and_metadata(self):
        document = Document(
            page_content="The service is available every weekday.",
            metadata={
                "source": "handbook",
                "page_start": 2,
                "page_end": 2,
                "section": "Availability",
                "subsection": "Schedule",
            },
        )

        context = build_context([(document, 0.2)])

        self.assertIn("handbook", context)
        self.assertIn("Availability", context)
        self.assertIn("The service is available every weekday.", context)


if __name__ == "__main__":
    unittest.main()
