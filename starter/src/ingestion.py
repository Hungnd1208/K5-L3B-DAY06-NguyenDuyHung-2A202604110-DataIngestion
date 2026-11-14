"""Read and normalize source data."""
import logging
import unicodedata
from pathlib import Path
from pypdf import PdfReader
from .models import DocumentPage

logger = logging.getLogger(__name__)


def get_chapter_by_page(page_number: int) -> str:
    """Map page number to chapter metadata based on document structure."""
    if page_number == 1:
        return "loi_mo_dau"
    elif 2 <= page_number <= 4:
        return "chuong_1"
    elif 5 <= page_number <= 6:
        return "chuong_2"
    elif 7 <= page_number <= 25:
        return "chuong_3"
    elif 26 <= page_number <= 28:
        return "chuong_4"
    elif 29 <= page_number <= 30:
        return "chuong_5"
    elif page_number == 31:
        return "ket_luan"
    else:
        return "tai_lieu_tham_khao"


def normalize_text(text: str) -> str:
    """Normalize Unicode (NFC), clean up carriage returns and inconsistent whitespace."""
    if not text:
        return ""
    # NFC normalization for consistent Vietnamese diacritics
    text = unicodedata.normalize("NFC", text)
    # Replace Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Clean up trailing spaces per line while preserving paragraphs
    lines = [line.strip() for line in text.split("\n")]
    # Remove excessive blank lines (>2 consecutive newlines)
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep page number, chapter, and source traceability.

    Detects and logs empty pages. Filters out non-content pages (e.g. references).
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    empty_pages: list[int] = []

    for idx, raw_page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = raw_page.extract_text() or ""
        normalized = normalize_text(raw_text)

        if not normalized:
            logger.warning("Empty or unextractable text on page %d of %s", page_num, path.name)
            empty_pages.append(page_num)
            continue

        chapter = get_chapter_by_page(page_num)
        is_content = (1 <= page_num <= 31)

        # Only retain valid content pages for ingestion (page 32 is References)
        if is_content:
            doc_page = DocumentPage(
                document_id=document_id,
                source=source,
                page=page_num,
                text=normalized,
                metadata={
                    "chapter": chapter,
                    "char_count": len(normalized),
                    "is_content_page": True
                }
            )
            pages.append(doc_page)
        else:
            logger.info("Page %d excluded as non-content (chapter=%s)", page_num, chapter)

    logger.info(
        "Successfully loaded %d content pages from %s (empty pages: %s)",
        len(pages), path.name, empty_pages
    )
    return pages
