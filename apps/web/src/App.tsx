import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "./config";
import "./App.css";

type Doc = {
  id: number;
  filename: string;
  content_type: string;
  size_bytes: number;
  created_at: string;
  stored_path?: string;
};

type Source = {
  ref: string;
  chunk_id: number;
  document_id: number;
  chunk_index: number;
  page: number | null;
  score: number;
  preview: string;
};

type AnswerResponse = {
  answer: string;
  confidence: string;
  citations: { ref: string; chunk_id: number; document_id: number }[];
  sources: Source[];
};

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function apiPostJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} failed: ${res.status}\n${text}`);
  }
  return res.json();
}

async function uploadDocument(file: File): Promise<Doc> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/documents`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed: ${res.status}\n${text}`);
  }
  return res.json();
}

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function App() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null);

  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);

  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedDoc = useMemo(
    () => docs.find((d) => d.id === selectedDocId) ?? null,
    [docs, selectedDocId]
  );

  async function refreshDocs() {
    setError(null);
    const data = await apiGet<Doc[]>("/documents");
    // newest first
    setDocs([...data].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)));
    if (selectedDocId === null && data.length > 0) setSelectedDocId(data[0].id);
  }

  useEffect(() => {
    refreshDocs().catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setAnswer(null);

    try {
      const doc = await uploadDocument(file);
      await refreshDocs();
      setSelectedDocId(doc.id);
    } catch (err) {
      setError(String(err));
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function runChunkEmbed(docId: number) {
    setError(null);
    setAnswer(null);
    try {
      await apiPostJson(`/documents/${docId}/chunk`, {});
      await apiPostJson(`/documents/${docId}/embed`, {});
    } catch (err) {
      setError(String(err));
      throw err;
    }
  }

  async function onAsk() {
    if (!selectedDocId) {
      setError("Select a document first.");
      return;
    }
    if (!question.trim()) return;

    setAsking(true);
    setError(null);
    setAnswer(null);

    try {
      // Ensure the doc is chunked/embedded (nice UX)
      await runChunkEmbed(selectedDocId);

      const data = await apiPostJson<AnswerResponse>("/answer", {
        query: question,
        top_k: 8,
        max_sources: 3,
        document_id: selectedDocId,
      });
      setAnswer(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="container">
      <header className="header">
        <div>
          <h1>Doc Detective</h1>
          <p className="sub">
            Upload a document, select it, and ask questions with citations.
          </p>
        </div>
        <div className="pill">API: {API_BASE}</div>
      </header>

      <div className="grid">
        <section className="card">
          <h2>Documents</h2>

          <div className="row">
            <label className="btn">
              {uploading ? "Uploading..." : "Upload"}
              <input
                type="file"
                onChange={onUpload}
                disabled={uploading}
                accept=".pdf,.docx,.txt"
                style={{ display: "none" }}
              />
            </label>

            <button className="btn secondary" onClick={() => refreshDocs()} disabled={uploading}>
              Refresh
            </button>
          </div>

          <div className="list">
            {docs.length === 0 ? (
              <div className="muted">No documents yet. Upload a PDF/DOCX/TXT.</div>
            ) : (
              docs.map((d) => (
                <button
                  key={d.id}
                  className={`listItem ${d.id === selectedDocId ? "active" : ""}`}
                  onClick={() => {
                    setSelectedDocId(d.id);
                    setAnswer(null);
                    setError(null);
                  }}
                >
                  <div className="title">{d.filename}</div>
                  <div className="meta">
                    <span>id {d.id}</span> · <span>{d.content_type}</span> ·{" "}
                    <span>{Math.round(d.size_bytes / 1024)} KB</span> ·{" "}
                    <span>{formatDate(d.created_at)}</span>
                  </div>
                </button>
              ))
            )}
          </div>

          {selectedDoc && (
            <div className="selected">
              <div className="muted">Selected</div>
              <div className="title">{selectedDoc.filename}</div>
              <div className="meta">document_id = {selectedDoc.id}</div>
            </div>
          )}
        </section>

        <section className="card">
          <h2>Ask</h2>

          <div className="row">
            <input
              className="input"
              placeholder="Ask a question about the selected document…"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") onAsk();
              }}
              disabled={asking}
            />
            <button className="btn primary" onClick={onAsk} disabled={asking || !question.trim()}>
              {asking ? "Working..." : "Ask"}
            </button>
          </div>

          <div className="hint">
            Tip: try “what company is this for”, “what’s the Python version”, “summarize the policy”.
          </div>

          {error && (
            <div className="error">
              <strong>Error</strong>
              <pre>{error}</pre>
            </div>
          )}

          {answer && (
            <div className="answer">
              <div className="answerTop">
                <div className="pill">confidence: {answer.confidence}</div>
              </div>

              <h3>Answer</h3>
              <pre className="answerText">{answer.answer}</pre>

              <h3>Sources</h3>
              <div className="sources">
                {answer.sources.map((s) => (
                  <div key={s.ref} className="source">
                    <div className="sourceHead">
                      <span className="pill">{s.ref}</span>
                      <span className="muted">
                        doc {s.document_id}
                        {s.page ? ` · p.${s.page}` : ""} · chunk {s.chunk_index} · score{" "}
                        {s.score.toFixed(3)}
                      </span>
                    </div>
                    <pre className="preview">{s.preview}</pre>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>

      <footer className="footer muted">
        Next: add auth, streaming answers, and a nicer citations UI.
      </footer>
    </div>
  );
}