import re

def extract_relevant_lines(query: str, text: str, max_lines: int = 4) -> str:
    """
    Very simple extractive heuristic:
    - split into lines
    - score lines by keyword overlap with query
    - return top N lines (keeps it grounded + concise)
    """
    q_words = set(re.findall(r"[a-z0-9']+", query.lower()))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    scored = []

    for ln in lines:
        w = set(re.findall(r"[a-z0-9']+", ln.lower()))
        score = len(q_words & w)
        if score > 0:
            scored.append((score, ln))

    if not scored:
        # fallback: first few non-empty lines
        return "\n".join(lines[:max_lines])

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [ln for _, ln in scored[:max_lines]]
    return "\n".join(top)