"""Implement fixed-size and structure-aware chunking."""
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split text into overlapping fixed-character windows and retain page metadata.

    Used for Strategy A (Baseline).
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be strictly less than size ({size})")

    text = page.text
    if not text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    idx = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + size, text_len)
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_id = f"{page.document_id}_p{page.page:02d}_fixed_{idx:03d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    metadata={
                        **page.metadata,
                        "strategy": "A",
                        "start_char": start,
                        "end_char": end,
                        "chunk_size": len(chunk_text),
                    }
                )
            )
            idx += 1

        if end == text_len:
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 700) -> list[Chunk]:
    """Group natural paragraphs without splitting mid-sentence, respecting max_size.

    Used for Strategy B & Strategy C.
    """
    text = page.text
    if not text:
        return []

    # Split by double newline or distinct paragraphs
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        # Fallback to single newline if no double newlines exist
        raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    grouped_blocks: list[str] = []
    current_block: list[str] = []
    current_len = 0

    for para in raw_paragraphs:
        para_len = len(para)

        # If a single paragraph is oversized, split it by sentence or newline
        if para_len > max_size:
            if current_block:
                grouped_blocks.append("\n\n".join(current_block))
                current_block = []
                current_len = 0

            # Subdivide oversized paragraph
            sub_sentences = [s.strip() for s in para.replace(". ", ".\n").split("\n") if s.strip()]
            sub_block: list[str] = []
            sub_len = 0
            for sent in sub_sentences:
                if sub_len + len(sent) + 1 <= max_size:
                    sub_block.append(sent)
                    sub_len += len(sent) + 1
                else:
                    if sub_block:
                        grouped_blocks.append(" ".join(sub_block))
                    sub_block = [sent]
                    sub_len = len(sent)
            if sub_block:
                grouped_blocks.append(" ".join(sub_block))
            continue

        if current_len + para_len + 2 <= max_size:
            current_block.append(para)
            current_len += para_len + 2
        else:
            if current_block:
                grouped_blocks.append("\n\n".join(current_block))
            current_block = [para]
            current_len = para_len

    if current_block:
        grouped_blocks.append("\n\n".join(current_block))

    chunks: list[Chunk] = []
    for idx, block in enumerate(grouped_blocks):
        block_text = block.strip()
        if not block_text:
            continue
        chunk_id = f"{page.document_id}_p{page.page:02d}_para_{idx:03d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=block_text,
                metadata={
                    **page.metadata,
                    "strategy": "B_C",
                    "chunk_size": len(block_text),
                    "paragraph_count": block.count("\n\n") + 1,
                }
            )
        )

    return chunks
