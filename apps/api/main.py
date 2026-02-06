import os
import hashlib
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, Mapped, mapped_column
from datetime import datetime, timezone

from db import SessionLocal, Base
from models import Document, Chunk
from chunking import chunk_text
from embeddings import embed_texts
from sqlalchemy import text, Boolean, String, DateTime, Integer
from pydantic import BaseModel
from extractive import extract_relevant_lines
from extractors import extract_text
from chunking import chunk_extracted

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: int | None = None

class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5
    max_sources: int = 3
    document_id: int | None = None

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # NEW: processing status fields
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

def confidence_from_distance(score: float) -> str:
    # If we're not using real embeddings, don't pretend precision
    if not os.getenv("OPENAI_API_KEY"):
        return "experimental"

    # Real embedding thresholds (tune later)
    if score <= 0.25:
        return "high"
    if score <= 0.5:
        return "medium"
    return "low"

app = FastAPI(title="Doc Detective API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/documents")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    # Save to disk
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    dest_path = STORAGE_DIR / safe_name

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    dest_path.write_bytes(contents)

    doc = Document(
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        stored_path=str(dest_path),
        size_bytes=len(contents),
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {
        "id": doc.id,
        "filename": doc.filename,
        "content_type": doc.content_type,
        "size_bytes": doc.size_bytes,
        "stored_path": doc.stored_path,
        "created_at": doc.created_at.isoformat(),
    }

@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "content_type": d.content_type,
            "size_bytes": d.size_bytes,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]

@app.post("/documents/{doc_id}/chunk")
def chunk_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    path = Path(doc.stored_path)
    if not path.exists():
        raise HTTPException(status_code=500, detail="Stored file missing")

    extracted = extract_text(path, doc.content_type)
    if not extracted:
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    chunk_pairs = chunk_extracted(extracted)

    # Clear existing chunks if reprocessing
    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()

    for idx, (page, content) in enumerate(chunk_pairs):
        db.add(
            Chunk(
                document_id=doc_id,
                chunk_index=idx,
                page=page,
                content=content,
            )
        )

    db.commit()
    return {"document_id": doc_id, "chunks_created": len(chunk_pairs)}

@app.get("/documents/{doc_id}/chunks")
def get_chunks(doc_id: int, db: Session = Depends(get_db)):
    chunks = (
        db.query(Chunk)
        .filter(Chunk.document_id == doc_id)
        .order_by(Chunk.chunk_index.asc())
        .all()
    )
    return [{"id": c.id, "chunk_index": c.chunk_index, "page": c.page, "content": c.content[:200]} for c in chunks]

@app.post("/documents/{doc_id}/embed")
def embed_document(doc_id: int, db: Session = Depends(get_db), batch_size: int = 50):
    chunks = (
        db.query(Chunk)
        .filter(Chunk.document_id == doc_id)
        .order_by(Chunk.chunk_index.asc())
        .all()
    )
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found for document")

    texts = [c.content for c in chunks]

    embedded = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embed_texts(batch)

        for j, vec in enumerate(vectors):
            chunks[i + j].embedding = vec
            embedded += 1

        db.commit()

    return {"document_id": doc_id, "chunks_embedded": embedded, "model": "text-embedding-3-small"}

@app.post("/search")
def search(req: SearchRequest, db: Session = Depends(get_db)):
    qvec = embed_texts([req.query])[0]

    # cosine distance: smaller is more similar
    # pgvector supports operators like <-> (L2) and <=> (cosine distance)
    results = (
        db.query(Chunk, (Chunk.embedding.cosine_distance(qvec)).label("score"))
        .filter(Chunk.embedding.isnot(None))
        .order_by("score")
        .limit(req.top_k)
        .all()
    )

    q = db.query(Chunk, (Chunk.embedding.cosine_distance(qvec)).label("score")).filter(Chunk.embedding.isnot(None))

    if req.document_id is not None:
        q = q.filter(Chunk.document_id == req.document_id)

    results = q.order_by("score").limit(req.top_k).all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "page": chunk.page,
            "score": float(score),
            "preview": chunk.content[:300],
        }
        for chunk, score in results
    ]

def _extract_company_from_text(text: str) -> str | None:
    """
    Heuristic for 'what company is this for' on cover letters:
    look for 'Dear' and grab the few lines before it (often contains company).
    Also tries to reconstruct line-broken company names in PDFs.
    """
    # normalize weird PDF whitespace
    cleaned = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]

    # try: lines before "Dear"
    for i, ln in enumerate(lines):
        if ln.lower().startswith("dear"):
            window = lines[max(0, i - 6): i]  # up to 6 lines before Dear
            # pick the most "name-like" line (letters/spaces, not phone/email/date)
            candidates = []
            for w in window:
                if any(x in w.lower() for x in ["@", "http", "linkedin", "github"]):
                    continue
                if re.search(r"\d", w):  # skip dates/addresses
                    continue
                if len(w) < 3:
                    continue
                candidates.append(w)
            if candidates:
                # PDFs often break company names across lines (e.g., Hudson / River / Trading)
                tail = candidates[-4:]  # up to last 4 lines before "Dear"
                joined = " ".join(tail)

                # clean up accidental double spaces
                joined = re.sub(r"\s{2,}", " ", joined).strip()

                # remove common header noise
                joined = re.sub(r"(?i)\bhiring manager\b", "", joined).strip()
                joined = re.sub(r"\s{2,}", " ", joined).strip()

                return joined

    # fallback: common cover-letter pattern "Hudson River Trading" may be line-broken;
    # just return the first capitalized phrase-like line we can find.
    for ln in lines[:40]:
        if re.fullmatch(r"[A-Za-z][A-Za-z .&'-]{2,}", ln) and len(ln.split()) >= 2:
            return ln

    return None

@app.post("/answer")
@app.post("/answer")
def answer(req: AnswerRequest, db: Session = Depends(get_db)):
    qvec = embed_texts([req.query])[0]

    # IMPORTANT: apply the document_id filter here (before ordering/limit)
    q = (
        db.query(Chunk, (Chunk.embedding.cosine_distance(qvec)).label("score"))
        .filter(Chunk.embedding.isnot(None))
    )
    if req.document_id is not None:
        q = q.filter(Chunk.document_id == req.document_id)

    results = (
        q.order_by("score")  # cosine distance: lower is better
         .limit(req.top_k)
         .all()
    )

    if not results:
        return {
            "answer": "No matching embedded chunks found. Upload → chunk → embed this document first.",
            "confidence": "experimental",
            "citations": [],
            "sources": [],
        }

    # Deduplicate by normalized content (collapses duplicates across docs/files)
    unique: list[tuple[Chunk, float]] = []
    seen: set[str] = set()

    for chunk, score in results:
        norm = " ".join(chunk.content.split())
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique.append((chunk, float(score)))

    top = unique[: max(1, min(req.max_sources, len(unique)))]
    best_chunk, best_score = top[0][0], float(top[0][1])

    # Special-case company questions to avoid weird line overlap issues on PDFs
    qlower = req.query.lower()
    if any(k in qlower for k in ["what company", "which company", "who is this for", "company is this", "cover letter for"]):
        company = _extract_company_from_text(best_chunk.content)
        snippet = company if company else extract_relevant_lines(req.query, best_chunk.content, max_lines=4)
    else:
        snippet = extract_relevant_lines(req.query, best_chunk.content, max_lines=4)

    # Build citations + sources (include page in citation text)
    citations = []
    sources = []
    for i, (chunk, score) in enumerate(top, start=1):
        citations.append({"ref": f"[{i}]", "chunk_id": chunk.id, "document_id": chunk.document_id})
        sources.append({
            "ref": f"[{i}]",
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "page": chunk.page,
            "score": float(score),
            "preview": chunk.content[:300],
        })

    pretty = []
    for s in sources:
        p = s["page"]
        if p is None:
            pretty.append(f'{s["ref"]} doc {s["document_id"]}')
        else:
            pretty.append(f'{s["ref"]} doc {s["document_id"]} p.{p}')
    pretty_citations = "; ".join(pretty)

    answer_text = f"{snippet}\n\nCitations: {pretty_citations}"

    return {
        "answer": answer_text,
        "confidence": confidence_from_distance(best_score),
        "citations": citations,
        "sources": sources,
    }