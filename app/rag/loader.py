import logging
from pathlib import Path
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_document(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if file_path.suffix.lower() == ".pdf":
        return _read_pdf(file_path)
    return _read_text(file_path)


def _read_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages).strip()
    logger.info(f"Loaded PDF: {file_path.name} ({len(reader.pages)} pages, {len(text)} chars)")
    return text


def _read_text(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8")
    logger.info(f"Loaded text: {file_path.name} ({len(text)} chars)")
    return text


import re

def split_into_chunks(text: str, chunk_size: int) -> list[str]:
    # First split by double newlines to catch obvious paragraph breaks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # Sub-split paragraphs that contain multiple numbered questions (e.g. "1. Question? Answer. 2. ...")
    # This regex looks for digits followed by a dot at the start of a clear sentence/block.
    sub_paragraphs = []
    for p in paragraphs:
        # Split by patterns like " 2. " or "10. " but try to keep the numbers
        # We look for " \d+\. " (with a space before) or start of string
        parts = re.split(r'(\s+\d+\.\s+)', p)
        if len(parts) > 1:
            current_p = parts[0]
            for i in range(1, len(parts), 2):
                num = parts[i]
                content = parts[i+1] if i+1 < len(parts) else ""
                sub_paragraphs.append(current_p.strip())
                current_p = num + content
            sub_paragraphs.append(current_p.strip())
        else:
            sub_paragraphs.append(p)
            
    # Remove empty or very short snippets
    sub_paragraphs = [p for p in sub_paragraphs if len(p) > 10]

    chunks: list[str] = []
    current = ""

    for paragraph in sub_paragraphs:
        # If the paragraph itself is larger than chunk_size, we just add it (embedding model will truncate if needed)
        # but try to keep it under chunk_size if possible.
        if len(current) + len(paragraph) <= chunk_size:
            current += (" " if current else "") + paragraph
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    logger.info(f"Split into {len(chunks)} chunks (chunk_size={chunk_size})")
    return chunks
