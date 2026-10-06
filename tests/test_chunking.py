"""Tests for layout-aware semantic chunking (src/ingestion/chunk.py)."""

import unittest

from src.ingestion.chunk import (
    build_semantic_units,
    is_bold_box,
    is_subheading,
    is_top_level_section,
    units_to_documents,
    validate_parsed_document,
)
from src.errors import ParsedDataError


def _make_box(text, x0=56.0, boxclass="text", header_level=0, bold=False):
    """Build a minimal PyMuPDF4LLM-style layout box for testing."""
    font = "AngsanaNew-Bold" if bold else "AngsanaNew"
    return {
        "x0": x0,
        "boxclass": boxclass,
        "header_level": header_level,
        "textlines": [
            {"spans": [{"text": text, "font": font}]}
        ],
    }


def _make_document(boxes, page_number=1):
    """Wrap boxes into a minimal parsed-document structure."""
    return {
        "pages": [
            {
                "page_number": page_number,
                "boxes": boxes,
            }
        ]
    }


class TestTopLevelSectionDetection(unittest.TestCase):
    def test_numbered_heading_at_left_margin(self):
        box = _make_box("7. Terms of Use", x0=56.0)
        self.assertTrue(is_top_level_section(box, "7. Terms of Use"))

    def test_numbered_heading_too_far_right(self):
        box = _make_box("7. Terms of Use", x0=120.0)
        self.assertFalse(is_top_level_section(box, "7. Terms of Use"))

    def test_unnumbered_text_is_not_section(self):
        box = _make_box("In the event of a cancer diagnosis", x0=56.0)
        self.assertFalse(is_top_level_section(box, "In the event of a cancer diagnosis"))


class TestSubheadingDetection(unittest.TestCase):
    def test_bullet_is_subheading(self):
        box = _make_box("• In the event of a cancer diagnosis", x0=90.0)
        self.assertTrue(is_subheading(box, "• In the event of a cancer diagnosis"))

    def test_section_header_class_indented(self):
        box = _make_box("Claim notification", x0=128.0, boxclass="section-header")
        self.assertTrue(is_subheading(box, "Claim notification"))

    def test_bold_short_text_indented(self):
        box = _make_box("Medical examination", x0=128.0, bold=True)
        self.assertTrue(is_subheading(box, "Medical examination"))

    def test_long_body_text_is_not_subheading(self):
        long_text = "document content " * 20
        box = _make_box(long_text, x0=128.0, bold=True)
        self.assertFalse(is_subheading(box, long_text))


class TestBoldDetection(unittest.TestCase):
    def test_bold_box(self):
        box = _make_box("heading text", bold=True)
        self.assertTrue(is_bold_box(box))

    def test_regular_box(self):
        box = _make_box("body text", bold=False)
        self.assertFalse(is_bold_box(box))

    def test_empty_textlines(self):
        box = {"textlines": None}
        self.assertFalse(is_bold_box(box))


class TestSemanticUnits(unittest.TestCase):
    def test_section_creates_unit(self):
        boxes = [
            _make_box("7. Terms of Use", x0=56.0),
            _make_box("Additional details", x0=74.0),
        ]
        data = _make_document(boxes)
        units = build_semantic_units(data, "test-doc")

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["section"], "7. Terms of Use")
        self.assertEqual(units[0]["source"], "test-doc")

    def test_subsection_prepends_parent_section(self):
        boxes = [
            _make_box("11. Illness notification", x0=56.0),
            _make_box("• In the event of a cancer diagnosis", x0=90.0),
            _make_box("Details...", x0=110.0),
        ]
        data = _make_document(boxes)
        units = build_semantic_units(data, "test-doc")

        # First unit: the section heading alone.
        # Second unit: the subsection with parent prepended.
        subsection_unit = units[-1]
        self.assertIn("11. Illness notification", subsection_unit["text"])
        self.assertEqual(subsection_unit["subsection"], "• In the event of a cancer diagnosis")

    def test_metadata_includes_page_numbers(self):
        boxes = [_make_box("content text", x0=74.0)]
        data = _make_document(boxes, page_number=5)
        units = build_semantic_units(data, "test-doc")

        self.assertEqual(units[0]["page_start"], 5)
        self.assertEqual(units[0]["page_end"], 5)


class TestUnitsToDocuments(unittest.TestCase):
    def test_documents_have_required_metadata(self):
        units = [
            {
                "text": "test content",
                "section": "1. Test",
                "subsection": None,
                "page_start": 1,
                "page_end": 1,
                "source": "test-doc",
            }
        ]
        docs = units_to_documents(units)

        self.assertGreater(len(docs), 0)
        meta = docs[0].metadata
        self.assertIn("source", meta)
        self.assertIn("page_start", meta)
        self.assertIn("section", meta)
        self.assertIn("chunk_index", meta)
        self.assertIn("unit_index", meta)


class TestValidation(unittest.TestCase):
    def test_empty_dict_raises(self):
        with self.assertRaises(ParsedDataError):
            validate_parsed_document({}, "broken.json")

    def test_missing_page_number_raises(self):
        data = {"pages": [{"boxes": []}]}
        with self.assertRaises(ParsedDataError):
            validate_parsed_document(data, "broken.json")

    def test_valid_document_passes(self):
        data = {"pages": [{"page_number": 1, "boxes": [{"textlines": []}]}]}
        validate_parsed_document(data, "ok.json")  # Should not raise.


if __name__ == "__main__":
    unittest.main()
