import pymupdf4llm

from ..errors import InputDataError
from ..settings import PARSED_DIR, PDF_DIR


def parse_pdfs() -> None:
    if not PDF_DIR.exists():
        raise InputDataError(
            f"PDF folder not found: {PDF_DIR}. Create it and add policy PDFs."
        )

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise InputDataError(
            f"No PDF files found in {PDF_DIR}. Add at least one policy PDF first."
        )

    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:

        print(f"\nParsing: {pdf_path.name}")

        markdown = pymupdf4llm.to_markdown(
            str(pdf_path)
        )

        md_path = PARSED_DIR / f"{pdf_path.stem}.md"

        md_path.write_text(
            markdown,
            encoding="utf-8"
        )

        # Layout JSON is consumed by the semantic chunker.
        json_text = pymupdf4llm.to_json(
            str(pdf_path)
        )

        json_path = PARSED_DIR / f"{pdf_path.stem}.json"

        json_path.write_text(
            json_text,
            encoding="utf-8"
        )

        print(f"Saved Markdown: {md_path}")
        print(f"Saved JSON:     {json_path}")


if __name__ == "__main__":
    parse_pdfs()
