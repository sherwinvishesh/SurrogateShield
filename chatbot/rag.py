"""
chatbot/rag.py — RAG Integration

Local Retrieval-Augmented Generation using ChromaDB and
sentence-transformers (all-MiniLM-L6-v2).

Design:
  - No server required — chromadb.Client() runs in-process
  - All documents anonymised via the SurrogateShield pipeline
    BEFORE being embedded and stored
  - Queries anonymised before retrieval
  - Retrieved context passed to Claude API with sanitised query
  - Claude's response run through ResolvePass before display

This module is in chatbot/ but RAG-specific pipeline logic
(anonymise → embed → query) is coordinated by pipeline.py.
"""

from __future__ import annotations

from typing import List, Optional

from config import EMBEDDING_MODEL, RAG_COLLECTION_NAME, RAG_CHUNK_SIZE, RAG_TOP_K
from util import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# RAGStore
# ─────────────────────────────────────────────

class RAGStore:
    """
    Local vector store backed by ChromaDB and sentence-transformers.

    All text stored in this index has been anonymised by the
    SurrogateShield pipeline before indexing. Queries are
    anonymised before retrieval.

    Attributes:
        _collection: ChromaDB collection for document storage.
        _model:      SentenceTransformer embedding model.
    """

    def __init__(self) -> None:
        """
        Initialise the ChromaDB client and embedding model.

        Both are loaded lazily — importation errors are raised
        at construction time with clear messages.
        """
        try:
            import chromadb
            # PersistentClient stores data to disk so indexed documents
            # survive process restarts. In-memory chromadb.Client() resets
            # every session — requiring add-doc to be re-run each time.
            self._client = chromadb.PersistentClient(path="./chroma_db")
            self._collection = self._client.get_or_create_collection(
                name=RAG_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"[RAG] ChromaDB collection '{RAG_COLLECTION_NAME}' ready "
                f"({self._collection.count()} documents)"
            )
        except ImportError:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            )
        except Exception as exc:
            raise RuntimeError(f"[RAG] Failed to initialise ChromaDB: {exc}") from exc

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"[RAG] Loaded embedding model: {EMBEDDING_MODEL}")
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as exc:
            raise RuntimeError(
                f"[RAG] Failed to load embedding model '{EMBEDDING_MODEL}': {exc}"
            ) from exc

        self._doc_counter: int = self._collection.count()

    def _embed(self, text: str) -> List[float]:
        """
        Embed a text string using the sentence-transformers model.

        Args:
            text: Text to embed.

        Returns:
            Embedding as a list of floats.
        """
        return self._model.encode(text, convert_to_numpy=True).tolist()

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

        Splits into chunks, embeds each chunk, and stores in ChromaDB.

        Args:
            sanitised_text: Anonymised document text (no real PII).
            metadata:       Optional metadata dict (e.g. filename, source).

        Returns:
            Number of chunks indexed.
        """
        chunks = self.chunk_text(sanitised_text)
        meta = metadata or {}

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"doc_{self._doc_counter}_{i}"
            ids.append(chunk_id)
            embeddings.append(self._embed(chunk))
            documents.append(chunk)
            metadatas.append({**meta, "chunk_index": i})
            self._doc_counter += 1

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                f"[RAG] Indexed {len(chunks)} chunks "
                f"(total: {self._collection.count()})"
            )
        except Exception as exc:
            logger.error(f"[RAG] Failed to add documents: {exc}")
            return 0

        return len(chunks)

    def query(self, sanitised_query: str, n: int = RAG_TOP_K) -> List[str]:
        """
        Retrieve the top-n most relevant chunks for a sanitised query.

        Args:
            sanitised_query: Query text with PII already replaced.
            n:               Number of chunks to return.

        Returns:
            List of document chunk strings, most relevant first.
        """
        if self._collection.count() == 0:
            logger.debug("[RAG] Collection is empty — returning no results")
            return []
        try:
            query_embedding = self._embed(sanitised_query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n, self._collection.count()),
                include=["documents"],
            )
            docs: List[str] = results["documents"][0] if results["documents"] else []
            logger.debug(f"[RAG] Retrieved {len(docs)} chunks for query")
            return docs
        except Exception as exc:
            logger.error(f"[RAG] Query failed: {exc}")
            return []

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