import os
import re
import hashlib
import numpy as np

# Optional OpenAI integration (only used if key exists)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "1536"))

_token_re = re.compile(r"[a-zA-Z0-9']+")


def _hash_embed(text: str, dims: int = EMBED_DIMS) -> list[float]:
    """
    Deterministic, local embedding:
    - tokenize
    - hash tokens into a fixed-size vector
    - L2 normalize

    Not as good as real embeddings, but perfect for building the pipeline.
    """
    tokens = _token_re.findall(text.lower())
    v = np.zeros(dims, dtype=np.float32)

    for t in tokens:
        # stable hash -> index
        h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h, "little") % dims
        v[idx] += 1.0

    # normalize
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v /= norm

    return v.astype(float).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    key = os.getenv("OPENAI_API_KEY")

    # If you ever add a key later, it will automatically use OpenAI
    if key and OpenAI is not None:
        client = OpenAI(api_key=key)
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in resp.data]

    # Fallback: local hashing embeddings
    return [_hash_embed(t) for t in texts]
