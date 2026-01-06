"""Semantic search utilities using Sentence Transformers and FAISS."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_CORPUS = [
    "FastAPI makes it easy to build web APIs in Python.",
    "Transformers provide state-of-the-art NLP models.",
    "Semantic search finds meaningfully similar text snippets.",
    "Docker simplifies packaging and deploying applications.",
    "FAISS enables efficient similarity search over vectors.",
]


@dataclass
class SemanticSearchEngine:
    """In-memory semantic search backed by FAISS."""

    model_name: str = "all-MiniLM-L6-v2"
    _model: SentenceTransformer = field(init=False, repr=False)
    _index: faiss.IndexFlatIP = field(init=False, repr=False)
    _corpus: List[str] = field(default_factory=list, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(self.model_name)
        dim = int(self._model.get_sentence_embedding_dimension())
        self._index = faiss.IndexFlatIP(dim)
        self.index_texts(DEFAULT_CORPUS)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into normalized embeddings."""

        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

    def index_texts(self, texts: List[str]) -> None:
        """Add new texts to the FAISS index."""

        if not texts:
            return
        vectors = self._encode(texts)
        with self._lock:
            self._index.add(vectors)
            self._corpus.extend(texts)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Return the top_k most semantically similar texts."""

        if not self._corpus:
            return []

        query_vec = self._encode([query])
        with self._lock:
            scores, indices = self._index.search(query_vec, top_k)
        results: List[str] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._corpus):
                continue
            results.append(self._corpus[idx])
        return results

    def add_text(self, texts: List[str]) -> None:
        """Dynamically add texts to the corpus."""

        self.index_texts(texts)

    def get_corpus_size(self) -> int:
        """Return the current size of the corpus."""

        with self._lock:
            return len(self._corpus)


def get_semantic_engine() -> SemanticSearchEngine:
    """Return a singleton semantic search engine instance."""

    # Simple module-level singleton pattern.
    if not hasattr(get_semantic_engine, "_engine"):
        get_semantic_engine._engine = SemanticSearchEngine()
    return get_semantic_engine._engine  # type: ignore[attr-defined]
