"""Helpers for the structured JSON produced from policy PDFs."""

from typing import Any


def get_box_text(box: dict[str, Any]) -> str:
    """Extract non-empty span text from a parser layout box."""
    texts = []

    for line in box.get("textlines") or []:
        for span in line.get("spans") or []:
            text = span.get("text", "").strip()
            if text:
                texts.append(text)

    return " ".join(texts).strip()
