import json
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_layout import get_box_text
from ..errors import ParsedDataError
from ..settings import PARSED_DIR


size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def is_bold_box(box):
    total = 0
    bold = 0

    textlines = box.get("textlines") or []

    for line in textlines:
        spans = line.get("spans") or []

        for span in spans:
            text = span.get("text", "").strip()

            if not text:
                continue

            total += 1

            font = span.get("font", "").lower()

            if "bold" in font:
                bold += 1

    if total == 0:
        return False

    return bold / total >= 0.5


def is_top_level_section(box, text):
    """
    Examples: ``1. Introduction`` or ``7. Terms and conditions``.
    """

    x = box.get("x0", 999)

    numbered_heading = re.match(
        r"^\d+\.\s+\S+",
        text
    )

    return (
        numbered_heading is not None
        and x <= 70
    )


def is_subheading(box, text):
    """
    Examples include bullet labels and bold, indented headings.
    """

    x = box.get("x0", 0)
    box_class = box.get("boxclass", "")

    if text.startswith("•"):
        return True

    if box_class == "section-header" and x > 70:
        return True

    # Some bold indented headings are emitted as plain text by the parser.
    if (
        x >= 115
        and len(text) <= 120
        and is_bold_box(box)
    ):
        return True

    return False


def validate_parsed_document(data, source_name):
    """Check the minimum JSON structure required for layout-aware chunking."""
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise ParsedDataError(
            f"Malformed parsed JSON in {source_name}: expected a 'pages' list."
        )

    for page_index, page in enumerate(data["pages"], start=1):
        if not isinstance(page, dict):
            raise ParsedDataError(
                f"Malformed parsed JSON in {source_name}: page {page_index} is invalid."
            )

        if "page_number" not in page or not isinstance(page.get("boxes"), list):
            raise ParsedDataError(
                f"Malformed parsed JSON in {source_name}: page {page_index} "
                "must contain page_number and boxes."
            )

        if not all(isinstance(box, dict) for box in page["boxes"]):
            raise ParsedDataError(
                f"Malformed parsed JSON in {source_name}: page {page_index} has an invalid box."
            )


def build_semantic_units(data, source_name):
    units = []

    current_section = None
    current_subsection = None

    current_text = []
    page_start = None
    page_end = None

    def flush():
        nonlocal current_text
        nonlocal page_start
        nonlocal page_end

        if not current_text:
            return

        units.append({
            "text": "\n".join(current_text).strip(),
            "section": current_section,
            "subsection": current_subsection,
            "page_start": page_start,
            "page_end": page_end,
            "source": source_name,
        })

        current_text = []
        page_start = None
        page_end = None

    for page in data["pages"]:
        page_number = page["page_number"]

        for box in page["boxes"]:
            text = get_box_text(box)

            if not text:
                continue

            if is_top_level_section(box, text):
                flush()

                current_section = text
                current_subsection = None

                current_text.append(text)

                page_start = page_number
                page_end = page_number

                continue

            if (
                current_section is not None
                and is_subheading(box, text)
            ):
                flush()

                current_subsection = text

                # Repeat the parent heading so the chunk retains section context.
                current_text.append(current_section)
                current_text.append(text)

                page_start = page_number
                page_end = page_number

                continue

            if page_start is None:
                page_start = page_number

            page_end = page_number

            current_text.append(text)

    flush()

    return units


def units_to_documents(units):
    documents = []

    for unit_index, unit in enumerate(units):

        metadata = {
            "source": unit["source"],
            "page_start": unit["page_start"],
            "page_end": unit["page_end"],
            "section": unit["section"] or "",
            "subsection": unit["subsection"] or "",
            "unit_index": unit_index,
        }

        doc = Document(
            page_content=unit["text"],
            metadata=metadata,
        )

        # Split only semantic units that still exceed the target size.
        split_docs = size_splitter.split_documents([doc])

        for chunk_index, split_doc in enumerate(split_docs):
            split_doc.metadata["chunk_index"] = chunk_index

            documents.append(split_doc)

    return documents


def load_documents():
    all_documents = []

    if not PARSED_DIR.exists():
        raise ParsedDataError(
            f"Parsed-data folder not found: {PARSED_DIR}. Run 'python -m src.main ingest' first."
        )

    json_files = sorted(PARSED_DIR.glob("*.json"))

    if not json_files:
        raise ParsedDataError(
            f"No parsed JSON files found in {PARSED_DIR}. Run 'python -m src.main ingest' first."
        )

    for json_path in json_files:
        print(f"Processing: {json_path.name}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ParsedDataError(
                f"Malformed JSON in {json_path.name}: {exc.msg}. Re-run ingestion."
            ) from exc

        validate_parsed_document(data, json_path.name)

        units = build_semantic_units(
            data,
            source_name=json_path.stem,
        )

        documents = units_to_documents(units)

        all_documents.extend(documents)

        print(
            f"  semantic units: {len(units)}"
            f" -> chunks: {len(documents)}"
        )

    return all_documents


if __name__ == "__main__":
    documents = load_documents()

    print("\n")
    print("=" * 100)
    print(f"TOTAL CHUNKS: {len(documents)}")
    print("=" * 100)

    for i, doc in enumerate(documents[:20]):
        print(f"\nCHUNK {i}")
        print("-" * 100)

        print("METADATA:")
        print(doc.metadata)

        print("\nCONTENT:")
        print(doc.page_content)

        print("=" * 100)
