# Doc Detective

Doc Detective lets you upload documents and ask questions with grounded answers that include citations to the exact source text.
It’s built as a production-style RAG app with an evaluation suite for quality regression and latency/cost tracking.

## Planned Features
- [ ] Document upload (PDF/DOCX/TXT/MD)
- [ ] Chunking + embeddings + pgvector retrieval
- [ ] Answers with citations + source highlighting
- [ ] Confidence panel (evidence strength + coverage)
- [ ] Evaluation suite (golden Q/A, citation grounding, latency + cost)

## Tech Stack
- FastAPI (Python)
- Postgres + pgvector
- Next.js (React)

## Local Development (WIP)
Coming soon.

## Roadmap
- MVP: upload → index → ask → cite
- Add auth + per-user isolation
- Add eval harness + CI gates
