import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from db import SessionLocal
from models import Document, Chunk
from chunking import chunk_text
from embeddings import embed_texts
from sqlalchemy import text
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

app = FastAPI(title="Doc Detective API")

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

    # MVP: only text files first
    if doc.content_type not in ("text/plain", "text/markdown"):
        raise HTTPException(status_code=400, detail=f"Unsupported content_type for MVP chunking: {doc.content_type}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = chunk_text(text)

    # Clear existing chunks if reprocessing
    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()

    for idx, content in enumerate(parts):
        db.add(Chunk(document_id=doc_id, chunk_index=idx, page=None, content=content))

    db.commit()

    return {"document_id": doc_id, "chunks_created": len(parts)}

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
