from __future__ import annotations
from typing import List, Tuple, Optional


def chunk_text(text: str, max_chars: int = 3000, overlap: int = 300) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def chunk_extracted(
    extracted: List[Tuple[Optional[int], str]],
    max_chars: int = 3000,
    overlap: int = 300,
) -> List[Tuple[Optional[int], str]]:
    """
    Takes [(page, text), ...] and returns [(page, chunk), ...] preserving page.
    """
    out: List[Tuple[Optional[int], str]] = []
    for page, text in extracted:
        for c in chunk_text(text, max_chars=max_chars, overlap=overlap):
            out.append((page, c))
    return out
