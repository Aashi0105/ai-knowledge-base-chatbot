"""Embeddings generation service for AI Knowledge Based Chatbot.

Provides local embedding generation using Sentence Transformers.
Model loading is performed lazily and cached once per process.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """Manages sentence-transformer embeddings with lazy initialization."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[Any] = None,
    ) -> None:
        """Initialize EmbeddingService.

        Args:
            model_name: Name or path of SentenceTransformer model. Defaults to
                environment variable EMBEDDING_MODEL or 'sentence-transformers/all-MiniLM-L6-v2'.
            model: Optional pre-initialized or mocked model instance. If provided,
                automatic loading is bypassed (useful for fast offline tests).
        """
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        self._model = model

    @property
    def model(self) -> Any:
        """Lazily load and return the SentenceTransformer model instance."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}': {e}"
                ) from e
        return self._model

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Generate normalized vector embeddings for a sequence of texts.

        Args:
            texts: List or sequence of text strings to embed.
            batch_size: Batch size for model inference.

        Returns:
            List of float vectors, preserving exact input ordering.

        Raises:
            ValueError: If texts contains invalid non-string entries.
        """
        if not texts:
            return []

        # Validate that all entries are strings
        cleaned_texts: list[str] = []
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Expected string at index {idx}, got {type(text).__name__}"
                )
            # Ensure non-empty string for embedding model stability
            cleaned_texts.append(text if text.strip() else " ")

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Convert numpy array or tensors to list of float lists
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(vec) for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Generate a normalized vector embedding for a single query text.

        Args:
            query: The natural language search query.

        Returns:
            Vector embedding as a list of floats.

        Raises:
            ValueError: If the query is empty or only whitespace.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace.")

        result = self.embed_texts([query.strip()])
        return result[0]

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        sample = self.embed_texts(["dimension_probe"])
        return len(sample[0])


_GLOBAL_EMBEDDING_SERVICE: Optional[EmbeddingService] = None


def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """Get or create the process-wide cached EmbeddingService singleton."""
    global _GLOBAL_EMBEDDING_SERVICE
    if _GLOBAL_EMBEDDING_SERVICE is None or (
        model_name and _GLOBAL_EMBEDDING_SERVICE.model_name != model_name
    ):
        _GLOBAL_EMBEDDING_SERVICE = EmbeddingService(model_name=model_name)
    return _GLOBAL_EMBEDDING_SERVICE
