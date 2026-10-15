"""Unit tests for EmbeddingService using mocked model (Phase 3)."""

import pytest
from src.embeddings import EmbeddingService


class MockTransformerModel:
    """Deterministic mock embedding model producing normalized fake vectors."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            seed = sum(ord(c) for c in text) % 50
            raw_vec = [(seed + i * 1.5) for i in range(self.dim)]
            norm = sum(x * x for x in raw_vec) ** 0.5 or 1.0
            normalized = [round(x / norm, 6) for x in raw_vec]
            results.append(normalized)
        return results


# 1. Text inputs produce vectors through a mocked embedding model.
def test_mocked_embeddings_generation():
    mock_model = MockTransformerModel(dim=16)
    service = EmbeddingService(model=mock_model)

    texts = ["Database indexing with B-Trees", "Distributed consensus with Paxos"]
    vectors = service.embed_texts(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) == 16
    assert len(vectors[1]) == 16
    assert isinstance(vectors[0][0], float)
    # Vectors should be deterministic and different for different texts
    assert vectors[0] != vectors[1]


# 2. Empty input is handled correctly.
def test_empty_embedding_inputs():
    mock_model = MockTransformerModel(dim=16)
    service = EmbeddingService(model=mock_model)

    assert service.embed_texts([]) == []

    with pytest.raises(ValueError) as exc1:
        service.embed_query("")
    assert "empty" in str(exc1.value).lower()

    with pytest.raises(ValueError) as exc2:
        service.embed_query("   ")
    assert "empty" in str(exc2.value).lower()


# 3. Query and document embedding functions use compatible dimensions.
def test_query_and_document_dimensions_match():
    mock_model = MockTransformerModel(dim=32)
    service = EmbeddingService(model=mock_model)

    query_vec = service.embed_query("What is virtual memory?")
    doc_vecs = service.embed_texts(["Virtual memory maps virtual addresses to physical pages."])

    assert len(query_vec) == 32
    assert len(doc_vecs[0]) == 32
    assert service.dimension == 32


# 4. Embedding errors are reported clearly.
def test_embedding_errors_reported():
    mock_model = MockTransformerModel(dim=16)
    service = EmbeddingService(model=mock_model)

    # Invalid input types in texts list
    with pytest.raises(ValueError) as exc_info:
        service.embed_texts([123, 456])  # type: ignore
    assert "Expected string" in str(exc_info.value)
