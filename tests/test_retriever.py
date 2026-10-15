"""Unit tests for semantic Retriever module (Phase 3)."""

from pathlib import Path
import pytest
from src.chunker import TextChunk
from src.embeddings import EmbeddingService
from src.retriever import RetrievedChunk, Retriever
from src.vector_store import VectorStore
from tests.test_embeddings import MockTransformerModel


@pytest.fixture
def test_retriever_setup(tmp_path: Path):
    """Fixture providing a configured Retriever with isolated ChromaDB and mock embedding model."""
    mock_model = MockTransformerModel(dim=8)
    embedding_service = EmbeddingService(model=mock_model)
    vector_store = VectorStore(persist_directory=str(tmp_path))
    retriever = Retriever(vector_store=vector_store, embedding_service=embedding_service)
    return retriever, vector_store, embedding_service


# 13. A query returns the stored matching chunks.
def test_query_returns_stored_matching_chunks(test_retriever_setup):
    retriever, vector_store, embedding_service = test_retriever_setup

    chunk1 = TextChunk(
        chunk_id="chunk_dbms",
        document_id="doc_dbms",
        source="dbms.pdf",
        page=2,
        chunk_index=0,
        text="Relational database management systems utilize SQL for queries.",
    )
    chunk2 = TextChunk(
        chunk_id="chunk_net",
        document_id="doc_net",
        source="networks.txt",
        page=None,
        chunk_index=0,
        text="The Transmission Control Protocol ensures reliable byte stream delivery.",
    )

    vec1 = embedding_service.embed_texts([chunk1.text])[0]
    vec2 = embedding_service.embed_texts([chunk2.text])[0]

    vector_store.upsert_chunks([chunk1, chunk2], [vec1, vec2])

    results = retriever.retrieve(query="SQL database queries", top_k=2)
    assert len(results) > 0
    assert any(r.chunk_id == "chunk_dbms" for r in results)


# 14. The top-k limit is respected.
def test_top_k_limit_respected(test_retriever_setup):
    retriever, vector_store, embedding_service = test_retriever_setup

    chunks = [
        TextChunk(
            chunk_id=f"c_{i}",
            document_id="doc1",
            source="paper.pdf",
            page=i + 1,
            chunk_index=i,
            text=f"Sample scientific paragraph number {i} discussing cloud architectures.",
        )
        for i in range(10)
    ]
    vecs = embedding_service.embed_texts([c.text for c in chunks])
    vector_store.upsert_chunks(chunks, vecs)

    res_k3 = retriever.retrieve("cloud architectures", top_k=3)
    assert len(res_k3) == 3

    res_k5 = retriever.retrieve("cloud architectures", top_k=5)
    assert len(res_k5) == 5


# 15. An empty query is handled correctly.
def test_empty_query_returns_empty_list(test_retriever_setup):
    retriever, _, _ = test_retriever_setup
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("valid query", top_k=0) == []


# 16. An empty collection returns no results.
def test_empty_collection_returns_no_results(test_retriever_setup):
    retriever, _, _ = test_retriever_setup
    results = retriever.retrieve("Any query into empty collection", top_k=4)
    assert results == []


# 17. Retrieved chunk IDs, text, and metadata remain intact.
def test_retrieved_chunk_metadata_integrity(test_retriever_setup):
    retriever, vector_store, embedding_service = test_retriever_setup

    chunk = TextChunk(
        chunk_id="chunk_integrity_check",
        document_id="doc_integrity",
        source="operating_systems.md",
        page=None,
        chunk_index=3,
        text="Process synchronization using semaphores and mutex locks.",
    )
    vec = embedding_service.embed_texts([chunk.text])[0]
    vector_store.upsert_chunks([chunk], [vec])

    results = retriever.retrieve("synchronization locks", top_k=1)
    assert len(results) == 1

    retrieved = results[0]
    assert retrieved.chunk_id == "chunk_integrity_check"
    assert retrieved.document_id == "doc_integrity"
    assert retrieved.source == "operating_systems.md"
    assert retrieved.page is None  # -1 restored to None
    assert retrieved.chunk_index == 3
    assert retrieved.text == chunk.text


# 18. Relevance or distance information is represented consistently.
def test_distance_and_similarity_metric_consistency(test_retriever_setup):
    retriever, vector_store, embedding_service = test_retriever_setup

    chunk = TextChunk(
        chunk_id="metric_chunk",
        document_id="doc_metric",
        source="algorithms.pdf",
        page=10,
        chunk_index=0,
        text="Dijkstra algorithm finds shortest paths in directed graphs.",
    )
    vec = embedding_service.embed_texts([chunk.text])[0]
    vector_store.upsert_chunks([chunk], [vec])

    results = retriever.retrieve("Dijkstra algorithm", top_k=1)
    assert len(results) == 1

    r = results[0]
    assert isinstance(r.distance, float)
    assert isinstance(r.similarity_score, float)
    assert r.distance >= 0.0
    # Similarity score should equal 1.0 - distance (or bounded)
    assert abs(r.similarity_score - (1.0 - r.distance)) < 1e-4


# 19. Score threshold filters out chunks below the similarity cutoff.
def test_score_threshold_filters_weak_chunks(test_retriever_setup):
    retriever, vector_store, embedding_service = test_retriever_setup

    chunk = TextChunk(
        chunk_id="chunk_cutoff",
        document_id="doc_cutoff",
        source="cutoff_test.txt",
        page=None,
        chunk_index=0,
        text="A test chunk for similarity cutoff verification.",
    )
    vec = embedding_service.embed_texts([chunk.text])[0]
    vector_store.upsert_chunks([chunk], [vec])

    # Unfiltered retrieval returns the chunk
    all_res = retriever.retrieve("test query", top_k=1, score_threshold=None)
    assert len(all_res) == 1
    actual_sim = all_res[0].similarity_score

    # Threshold higher than actual similarity must filter it out
    filtered_res = retriever.retrieve("test query", top_k=1, score_threshold=actual_sim + 0.1)
    assert len(filtered_res) == 0

    # Threshold lower than or equal to actual similarity must keep it
    kept_res = retriever.retrieve("test query", top_k=1, score_threshold=actual_sim - 0.01)
    assert len(kept_res) == 1

