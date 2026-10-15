"""Persistent vector storage module using ChromaDB for AI Knowledge Based Chatbot.

Manages persistent storage, upsertion, querying, and deletion of text chunks
and their corresponding vector embeddings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv

from src.chunker import TextChunk

load_dotenv()

DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "academic_knowledge_base"


class VectorStore:
    """Interface for persistent ChromaDB vector storage."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: Optional[ClientAPI] = None,
    ) -> None:
        """Initialize ChromaDB client and collection.

        Args:
            persist_directory: Local directory for ChromaDB persistence. Defaults
                to environment variable CHROMA_DB_PATH or './chroma_db'.
            collection_name: Name of ChromaDB collection. Defaults to 'academic_knowledge_base'.
            client: Optional pre-configured ClientAPI instance (useful for testing).
        """
        self.persist_directory = str(
            Path(
                persist_directory
                or os.getenv("CHROMA_DB_PATH")
                or DEFAULT_CHROMA_PATH
            ).resolve()
        )
        self.collection_name = collection_name

        if client is not None:
            self._client = client
        else:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)

        self._collection: Optional[Collection] = None

    @property
    def collection(self) -> Collection:
        """Get or create the target collection with cosine distance."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert_chunks(
        self,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Insert or update chunks and their embeddings in the vector store.

        Args:
            chunks: Sequence of TextChunk objects to store.
            embeddings: Corresponding embedding vectors for each chunk.

        Returns:
            Number of chunks indexed.

        Raises:
            ValueError: If chunk and embedding sequence lengths do not match.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embedding_list: list[list[float]] = []

        for chunk, emb in zip(chunks, embeddings):
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            embedding_list.append(list(emb))
            metadatas.append(
                {
                    "document_id": str(chunk.document_id),
                    "source": str(chunk.source),
                    "page": int(chunk.page) if chunk.page is not None else -1,
                    "chunk_index": int(chunk.chunk_index),
                }
            )

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embedding_list,
            metadatas=metadatas,
        )
        return len(ids)

    def query_similar(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Query the collection for chunks closest to the query embedding.

        Args:
            query_embedding: The vector representation of the query.
            top_k: Maximum number of nearest chunks to retrieve.
            where: Optional ChromaDB metadata filter dictionary.

        Returns:
            Dictionary containing 'ids', 'documents', 'metadatas', and 'distances'.
        """
        if self.collection.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Cap top_k to the number of indexed records
        n_results = min(top_k, self.collection.count())
        if n_results <= 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_params: dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        return self.collection.query(**query_params)

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document ID.

        Args:
            document_id: The document identifier to remove.

        Returns:
            Number of records removed.
        """
        if not document_id:
            return 0

        # Find count of chunks matching document_id before deletion
        matches = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        count = len(matches["ids"]) if matches and "ids" in matches else 0

        if count > 0:
            self.collection.delete(where={"document_id": document_id})

        return count

    def has_document(self, document_id: str) -> bool:
        """Check if any chunks for the given document_id are indexed."""
        if not document_id or self.collection.count() == 0:
            return False
        results = self.collection.get(
            where={"document_id": document_id},
            limit=1,
            include=["metadatas"],
        )
        return bool(results and results.get("ids"))

    def get_stats(self) -> dict[str, Any]:
        """Return basic statistics about the current index."""
        total_chunks = self.collection.count()
        doc_ids: set[str] = set()

        if total_chunks > 0:
            all_records = self.collection.get(include=["metadatas"])
            for meta in all_records.get("metadatas", []):
                if meta and "document_id" in meta:
                    doc_ids.add(meta["document_id"])

        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_chunks": total_chunks,
            "total_documents": len(doc_ids),
            "document_ids": sorted(list(doc_ids)),
        }

    def reset_collection(self) -> None:
        """Delete and recreate the current collection."""
        self._client.delete_collection(name=self.collection_name)
        self._collection = None
