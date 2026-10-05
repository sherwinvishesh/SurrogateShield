# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""
chatbot/rag.py — RAG Integration

Local Retrieval-Augmented Generation using ChromaDB and
sentence-transformers (all-MiniLM-L6-v2).

Design:
  - No server required — chromadb runs in-process; telemetry off
  - All documents anonymised via the SurrogateShield pipeline
    BEFORE being embedded and stored
  - Queries anonymised before retrieval
  - Retrieved context passed to Claude API with sanitised query
  - Claude's response run through ResolvePass before display

This module is in chatbot/ but RAG-specific pipeline logic
(anonymise → embed → query) is coordinated by pipeline.py.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from config import EMBEDDING_MODEL, RAG_CHUNK_SIZE, RAG_COLLECTION_NAME, RAG_DIR, RAG_TOP_K
from surrogateshield.core.storage.shadow_map import PathLike, ensure_private_dir, home
from util import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# RAGStore
# ─────────────────────────────────────────────

class RAGError(RuntimeError):
    """The vector store could not index or delete documents."""


def rag_dir() -> Path:
    """Directory of the chroma index (private, under the SurrogateShield home)."""
    if RAG_DIR:
        return Path(RAG_DIR).expanduser().resolve()
    return home() / "rag"


class RAGStore:
    """
    Local vector store backed by ChromaDB and sentence-transformers.

    All text stored in this index has been anonymised by the
    SurrogateShield pipeline before indexing; the surrogate → original pairs
    live in the encrypted ``rag_global`` shadow map (pipeline.py), never in
    the index. ChromaDB telemetry is off (audit I16).

    Args:
        path:     Index directory (default :func:`rag_dir`, mode 0700).
        embedder: ``text -> list[float]``; defaults to the sentence-transformers
                  model ``EMBEDDING_MODEL`` (tests pass a stub).
    """

    def __init__(self, path: Optional[PathLike] = None, *,
                 embedder: Optional[Callable[[str], List[float]]] = None) -> None:
        directory = ensure_private_dir(Path(path) if path is not None else rag_dir())
        if path is None and Path("chroma_db").is_dir():
            logger.warning(
                f"[RAG] ./chroma_db (pre-v2 location) is no longer used; the index "
                f"now lives in {directory}. Re-run add-doc for those documents.")
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is not installed. Run: pip install chromadb") from None
        try:
            self._client = chromadb.PersistentClient(
                path=str(directory), settings=Settings(anonymized_telemetry=False))
            self._collection = self._client.get_or_create_collection(
                name=RAG_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        except Exception as exc:
            raise RAGError(f"Failed to open the RAG index at {directory}: {exc}") from exc
        logger.info(f"[RAG] collection '{RAG_COLLECTION_NAME}' ready "
                    f"({self._collection.count()} chunks)")

        if embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("sentence-transformers is not installed. "
                                  "Run: pip install sentence-transformers") from None
            try:
                model = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as exc:
                raise RAGError(f"Failed to load embedding model '{EMBEDDING_MODEL}': {exc}") from exc
            embedder = lambda text: model.encode(text, convert_to_numpy=True).tolist()  # noqa: E731
        self._embed = embedder

    @staticmethod
    def chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE) -> List[str]:
        """
        Split *text* into overlapping chunks for indexing.

        Simple character-based chunking with 20% overlap.

        Args:
            text:       Full document text.
            chunk_size: Target chunk size in characters.

        Returns:
            List of text chunks.
        """
        if len(text) <= chunk_size:
            return [text]
        overlap = max(50, chunk_size // 5)
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def add_document(self, sanitised_text: str, metadata: Optional[dict] = None) -> int:
        """
        Index a document that has already been anonymised by the pipeline.

        Every chunk carries ``doc_id`` (random) and ``source`` metadata so the
        document can be removed with :meth:`forget`.

        Returns:
            Number of chunks indexed.

        Raises:
            RAGError: the index rejected the chunks.
        """
        meta = dict(metadata or {})
        doc_id = meta.setdefault("doc_id", uuid.uuid4().hex[:12])
        meta.setdefault("source", doc_id)
        chunks = [c for c in self.chunk_text(sanitised_text) if c.strip()]
        if not chunks:
            return 0
        try:
            self._collection.add(
                ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
                embeddings=[self._embed(c) for c in chunks],
                documents=chunks,
                metadatas=[{**meta, "chunk_index": i} for i in range(len(chunks))],
            )
        except Exception as exc:
            raise RAGError(f"Failed to index document {meta['source']!r}: {exc}") from exc
        logger.info(f"[RAG] Indexed {len(chunks)} chunks (total: {self._collection.count()})")
        return len(chunks)

    def query(self, sanitised_query: str, n: int = RAG_TOP_K) -> List[str]:
        """
        Retrieve the top-n most relevant chunks for a sanitised query.

        Args:
            sanitised_query: Query text with PII already replaced.
            n:               Number of chunks to return.

        Returns:
            List of document chunk strings, most relevant first (empty when
            the index is empty or the query fails — logged).
        """
        count = self._collection.count()
        if count == 0:
            logger.debug("[RAG] Collection is empty — returning no results")
            return []
        try:
            results = self._collection.query(
                query_embeddings=[self._embed(sanitised_query)],
                n_results=min(n, count), include=["documents"])
        except Exception as exc:
            logger.error(f"[RAG] Query failed: {exc}")
            return []
        docs: List[str] = results["documents"][0] if results["documents"] else []
        logger.debug(f"[RAG] Retrieved {len(docs)} chunks for query")
        return docs

    def documents(self) -> Dict[str, dict]:
        """``doc_id -> {"source", "chunks"}`` for every indexed document."""
        out: Dict[str, dict] = {}
        for meta in self._collection.get(include=["metadatas"])["metadatas"] or []:
            meta = meta or {}
            doc_id = meta.get("doc_id", "?")
            row = out.setdefault(doc_id, {"source": meta.get("source", doc_id), "chunks": 0})
            row["chunks"] += 1
        return out

    def texts(self) -> List[str]:
        """Every indexed chunk (sanitised text)."""
        return list(self._collection.get(include=["documents"])["documents"] or [])

    def forget(self, doc: str) -> int:
        """Remove every chunk of the document whose ``doc_id`` or ``source``
        is *doc*. Returns the number of chunks removed.

        Raises:
            RAGError: the index could not delete them.
        """
        ids = []
        for field in ("doc_id", "source"):
            ids += self._collection.get(where={field: doc}, include=[])["ids"]
        ids = sorted(set(ids))
        if ids:
            try:
                self._collection.delete(ids=ids)
            except Exception as exc:
                raise RAGError(f"Failed to delete {doc!r}: {exc}") from exc
        return len(ids)

    def build_context_prompt(self, chunks: List[str]) -> str:
        """
        Format retrieved chunks into a context block for Claude.

        Args:
            chunks: Retrieved document chunks.

        Returns:
            Formatted context string to prepend to the user message.
        """
        if not chunks:
            return ""
        formatted = "\n\n---\n\n".join(
            f"[Document excerpt {i + 1}]\n{chunk}"
            for i, chunk in enumerate(chunks)
        )
        return (
            f"Use the following retrieved context to help answer the question.\n\n"
            f"{formatted}\n\n---\n\nQuestion: "
        )

    def document_count(self) -> int:
        """Return total number of chunks in the collection."""
        return self._collection.count()
