"""Semantic retrieval module for AI Knowledge Based Chatbot.

Retrieves top-k relevant document chunks from ChromaDB for natural-language queries
using local vector similarity. Does NOT generate LLM answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence
from src.embeddings import EmbeddingService, get_embedding_service
from src.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a retrieved chunk with source metadata and similarity metric.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        document_id: Identifier of the parent document.
        source: Filename of the source document.
        page: 1-based page number for PDFs, or None for TXT/Markdown.
        text: Original chunk text content.
        chunk_index: 0-based chunk index in the document.
        distance: Raw ChromaDB distance (cosine distance, where 0.0 is identical).
        similarity_score: Derived similarity indicator (1.0 - distance), not a probability.
    """
    chunk_id: str
    document_id: str
    source: str
    page: Optional[int]
    text: str
    chunk_index: int
    distance: float
    similarity_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "page": self.page,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "distance": self.distance,
            "similarity_score": self.similarity_score,
        }


class Retriever:
    """Orchestrates query embedding and semantic search across indexed chunks."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        """Initialize Retriever.

        Args:
            vector_store: VectorStore instance for ChromaDB access.
            embedding_service: EmbeddingService instance for query vectorization.
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or get_embedding_service()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        where: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """Search ChromaDB for chunks semantically relevant to the user query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of relevant chunks to retrieve (default: 4).
            score_threshold: Optional minimum similarity score threshold (1.0 - cosine_distance).
                Chunks with similarity below this threshold are omitted.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of RetrievedChunk objects ordered by relevance. Returns empty list
            if query is empty or index contains no matching chunks.
        """
        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        # Generate normalized embedding vector for query
        query_vector = self.embedding_service.embed_query(query.strip())

        # Perform nearest-neighbor similarity search
        raw_results = self.vector_store.query_similar(
            query_embedding=query_vector,
            top_k=top_k,
            where=where,
        )

        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        retrieved_chunks: list[RetrievedChunk] = []

        for i, chunk_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = float(distances[i]) if i < len(distances) else 0.0

            # Cosine similarity metric: 1.0 - distance
            # Note: Chroma cosine distance is in range [0, 2]; this is a relative rank metric,
            # not a probability or confidence value.
            sim = round(max(0.0, 1.0 - dist), 4)

            if score_threshold is not None and sim < score_threshold:
                continue

            raw_page = meta.get("page", -1)
            restored_page: Optional[int] = (
                None if raw_page is None or int(raw_page) == -1 else int(raw_page)
            )

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", "unknown"),
                    source=meta.get("source", "unknown"),
                    page=restored_page,
                    text=text,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    distance=round(dist, 4),
                    similarity_score=sim,
                )
            )

        return retrieved_chunks
