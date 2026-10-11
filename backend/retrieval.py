"""Local semantic retrieval for Willow's library documents.

The index is built by scripts/build_index.py and stored on disk. No hosted
embedding or generation API is used. The embedding model is downloaded once,
then loaded from the local Hugging Face cache for subsequent offline runs.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
INDEX_DIR = BASE_DIR / "rag_index"
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = max(3, int(os.getenv("RAG_TOP_K", "4")))
RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.30"))
CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 35
QUERY_CACHE_SIZE = 128

_model = None
_model_lock = threading.Lock()
_index_lock = threading.Lock()
_query_cache: "OrderedDict[str, tuple[list[dict], float]]" = OrderedDict()
_cache_lock = threading.Lock()


@dataclass
class Chunk:
    chunk_id: str
    source: str
    title: str
    text: str
    start_word: int
    end_word: int


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    title: str
    text: str
    score: float


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, source: str, title: str, chunk_words: int = CHUNK_WORDS,
               overlap_words: int = CHUNK_OVERLAP_WORDS) -> List[Chunk]:
    """Split text into overlapping word windows while preserving source metadata."""
    cleaned = _clean_text(text)
    words = cleaned.split()
    if not words:
        return []
    if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Require chunk_words > overlap_words >= 0")
    step = chunk_words - overlap_words
    chunks: List[Chunk] = []
    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        if not part:
            break
        idx = len(chunks) + 1
        chunks.append(Chunk(
            chunk_id=f"{Path(source).stem}-{idx:03d}",
            source=source,
            title=title,
            text=" ".join(part),
            start_word=start,
            end_word=start + len(part),
        ))
        if start + chunk_words >= len(words):
            break
    return chunks


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def read_document_chunks(documents_dir: Path = DOCUMENTS_DIR) -> List[Chunk]:
    chunks: List[Chunk] = []
    files = sorted(p for p in documents_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".html", ".htm"})
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # The current corpus uses Markdown/plain text. Remove simple HTML tags for HTML input.
        if path.suffix.lower() in {".html", ".htm"}:
            raw = re.sub(r"<[^>]+>", " ", raw)
        rel = path.relative_to(documents_dir).as_posix()
        title = _title_from_text(raw, path.stem.replace("_", " ").title())
        chunks.extend(chunk_text(raw, rel, title))
    return chunks


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "The embedding dependency is missing. Run: pip install -r requirements.txt"
                    ) from exc
                _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def encode_texts(texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_index(documents_dir: Path = DOCUMENTS_DIR, index_dir: Path = INDEX_DIR) -> dict:
    chunks = read_document_chunks(documents_dir)
    if not chunks:
        raise RuntimeError(f"No supported documents found under {documents_dir}")
    vectors = encode_texts([c.text for c in chunks])
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", vectors)
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "document_count": len(list(p for p in documents_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".html", ".htm"})),
        "chunk_count": len(chunks),
        "chunk_words": CHUNK_WORDS,
        "overlap_words": CHUNK_OVERLAP_WORDS,
        "built_at_unix": time.time(),
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_index():
    embeddings_path = INDEX_DIR / "embeddings.npy"
    chunks_path = INDEX_DIR / "chunks.json"
    if not embeddings_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "The RAG index is missing. Run `python scripts/build_index.py` from the project root."
        )
    with _index_lock:
        # Small index: reading once and keeping it in process memory is inexpensive.
        if not hasattr(_load_index, "data"):
            vectors = np.load(embeddings_path).astype(np.float32)
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            if len(vectors) != len(chunks):
                raise RuntimeError("Index is inconsistent: embedding and chunk counts differ. Rebuild it.")
            _load_index.data = (vectors, chunks)
        return _load_index.data


def rank_vectors(query_vector: np.ndarray, embeddings: np.ndarray, chunks: Sequence[dict], top_k: int = TOP_K) -> List[RetrievedChunk]:
    """Pure cosine-ranking helper, also unit-testable without downloading a model."""
    if len(embeddings) == 0 or not chunks:
        return []
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    matrix = np.asarray(embeddings, dtype=np.float32)
    # Index and query vectors are normalised by SentenceTransformer; this is cosine similarity.
    scores = matrix @ q
    order = np.argsort(scores)[::-1][:max(0, top_k)]
    return [RetrievedChunk(
        chunk_id=chunks[int(i)]["chunk_id"],
        source=chunks[int(i)]["source"],
        title=chunks[int(i)]["title"],
        text=chunks[int(i)]["text"],
        score=float(scores[int(i)]),
    ) for i in order]


def retrieve(query: str, top_k: int = TOP_K, threshold: float = RELEVANCE_THRESHOLD) -> dict:
    """Return ranked matches, filtering weak matches and caching repeated queries."""
    query = query.strip()
    if not query:
        return {"matches": [], "latency_ms": 0.0, "cache_hit": False, "error": None}
    cache_key = f"{top_k}|{threshold:.3f}|{query.casefold()}"
    with _cache_lock:
        cached = _query_cache.get(cache_key)
        if cached:
            _query_cache.move_to_end(cache_key)
            matches, original_latency = cached
            return {"matches": [dict(m) for m in matches], "latency_ms": 0.0, "cache_hit": True, "error": None}
    started = time.perf_counter()
    try:
        vectors, chunks = _load_index()
        query_vector = encode_texts([query])[0]
        ranked = rank_vectors(query_vector, vectors, chunks, top_k=max(3, top_k))
        relevant = [asdict(item) for item in ranked if item.score >= threshold]
        elapsed_ms = (time.perf_counter() - started) * 1000
        with _cache_lock:
            _query_cache[cache_key] = (relevant, elapsed_ms)
            _query_cache.move_to_end(cache_key)
            while len(_query_cache) > QUERY_CACHE_SIZE:
                _query_cache.popitem(last=False)
        return {"matches": relevant, "latency_ms": elapsed_ms, "cache_hit": False, "error": None}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {"matches": [], "latency_ms": elapsed_ms, "cache_hit": False, "error": str(exc)}


def format_context(matches: Sequence[dict], max_chars: int = 4200) -> str:
    """Build a bounded context block; preserve the highest-ranked evidence first."""
    sections = []
    used = 0
    for match in matches:
        block = (f"[Source: {match['title']} | file: {match['source']} | "
                 f"similarity: {match['score']:.3f}]\n{match['text']}\n")
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining > 160:
                block = block[:remaining - 20] + "\n[chunk truncated]\n"
            else:
                break
        sections.append(block)
        used += len(block)
    return "\n".join(sections)


def format_sources(matches: Sequence[dict]) -> str:
    seen = set()
    labels = []
    for match in matches:
        key = (match["title"], match["source"])
        if key not in seen:
            seen.add(key)
            labels.append(f"{match['title']} ({match['source']})")
    if not labels:
        return ""
    return "\n\nSources: " + "; ".join(labels)
