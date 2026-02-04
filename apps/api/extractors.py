from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

from pypdf import PdfReader
import docx

# Returns list of (page, text). page is 1-based for PDFs; None when not applicable.
Extracted = List[Tuple[Optional[int], str]]


def extract_text(path: Path, content_type: str) -> Extracted:
    """
    Route to the right extractor based on content_type and/or extension.
    """
    ct = (content_type or "").lower()
    ext = path.suffix.lower()

    if ct == "application/pdf" or ext == ".pdf":
        return extract_pdf(path)

    if ct in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ) or ext in (".docx", ".doc"):
        # We'll handle .docx here. (Old .doc may fail; that's okay for MVP.)
        return extract_docx(path)

    # fallback: treat as text
    return extract_txt(path)


def extract_txt(path: Path) -> Extracted:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(None, text)] if text else []


def extract_pdf(path: Path) -> Extracted:
    reader = PdfReader(str(path))
    out: Extracted = []

    for i, page in enumerate(reader.pages, start=1):  # 1-based page numbers
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            out.append((i, text))

    return out


def extract_docx(path: Path) -> Extracted:
    d = docx.Document(str(path))

    # Simple: join paragraphs into one blob
    paras = [p.text.strip() for p in d.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paras).strip()

    return [(None, text)] if text else []