"""Unit tests for ChromaDB VectorStore (Phase 3)."""

from pathlib import Path
import pytest
from src.chunker import TextChunk
from src.vector_store import VectorStore


def make_sample_chunk(
    chunk_id: str = "doc1_p1_c0",
    doc_id: str = "doc1",
    source: str = "network_layers.pdf",
    page: int | None = 1,
    text: str = "The transport layer provides host-to-host communication services.",
    index: int = 0,
) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        source=source,
        page=page,
        chunk_index=index,
        text=text,
    )


# 5. A chunk can be inserted and retrieved.
def test_insert_and_retrieve_chunk(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))
    chunk = make_sample_chunk()
    embedding = [0.1, 0.2, 0.3, 0.4]

    inserted = store.upsert_chunks([chunk], [embedding])
    assert inserted == 1

    # Query with identical vector
    results = store.query_similar(query_embedding=embedding, top_k=1)
    ids = results["ids"][0]
    assert len(ids) == 1
    assert ids[0] == "doc1_p1_c0"
    assert results["documents"][0][0] == chunk.text


# 6. Source filename and page metadata are preserved.
def test_metadata_preservation(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))
    chunk = make_sample_chunk(source="dbms_acid.pdf", page=5, index=2)
    embedding = [0.5, 0.5, 0.0, 0.0]

    store.upsert_chunks([chunk], [embedding])
    results = store.query_similar(query_embedding=embedding, top_k=1)

    meta = results["metadatas"][0][0]
    assert meta["source"] == "dbms_acid.pdf"
    assert meta["page"] == 5
    assert meta["chunk_index"] == 2
    assert meta["document_id"] == "doc1"


# 7. TXT/Markdown chunks with page=None are restored correctly.
def test_txt_markdown_page_none_handling(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))
    txt_chunk = make_sample_chunk(
        chunk_id="txt_doc_c0",
        doc_id="txt_doc",
        source="notes.txt",
        page=None,
        text="Plain text document notes.",
    )
    embedding = [0.0, 0.8, 0.2, 0.0]

    store.upsert_chunks([txt_chunk], [embedding])
    results = store.query_similar(query_embedding=embedding, top_k=1)

    meta = results["metadatas"][0][0]
    assert meta["page"] == -1  # Encoded as -1 in storage


# 8. Upserting identical chunk IDs does not duplicate records.
def test_idempotent_upsert_no_duplicates(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))
    chunk = make_sample_chunk(chunk_id="chunk_stable_001")
    embedding = [0.2, 0.3, 0.4, 0.5]

    # Insert twice
    store.upsert_chunks([chunk], [embedding])
    assert store.collection.count() == 1

    store.upsert_chunks([chunk], [embedding])
    assert store.collection.count() == 1


# 9. Deleting a document removes only its chunks.
def test_delete_document_removes_only_target(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))

    chunk_a1 = make_sample_chunk(chunk_id="a1", doc_id="doc_A", text="Doc A chunk 1")
    chunk_a2 = make_sample_chunk(chunk_id="a2", doc_id="doc_A", text="Doc A chunk 2")
    chunk_b1 = make_sample_chunk(chunk_id="b1", doc_id="doc_B", text="Doc B chunk 1")

    vec = [0.1, 0.2, 0.3, 0.4]
    store.upsert_chunks([chunk_a1, chunk_a2, chunk_b1], [vec, vec, vec])
    assert store.collection.count() == 3

    # Delete doc_A
    removed = store.delete_document("doc_A")
    assert removed == 2
    assert store.collection.count() == 1

    # Remaining item must be from doc_B
    remaining = store.collection.get()
    assert remaining["ids"] == ["b1"]


# 10. Index statistics report the expected document and chunk counts.
def test_index_statistics_reporting(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))

    chunk1 = make_sample_chunk(chunk_id="c1", doc_id="doc1")
    chunk2 = make_sample_chunk(chunk_id="c2", doc_id="doc1")
    chunk3 = make_sample_chunk(chunk_id="c3", doc_id="doc2")

    store.upsert_chunks([chunk1, chunk2, chunk3], [[0.1]*4, [0.2]*4, [0.3]*4])

    stats = store.get_stats()
    assert stats["total_chunks"] == 3
    assert stats["total_documents"] == 2
    assert sorted(stats["document_ids"]) == ["doc1", "doc2"]
    assert store.has_document("doc1") is True
    assert store.has_document("doc3") is False


# 11. An empty indexing request is handled safely.
def test_empty_indexing_request_is_safe(tmp_path: Path):
    store = VectorStore(persist_directory=str(tmp_path))
    assert store.upsert_chunks([], []) == 0
    assert store.delete_document("") == 0


# 12. A persistent index can be reopened from the same temporary directory.
def test_reopen_persistent_index(tmp_path: Path):
    db_path = str(tmp_path / "chroma_test_store")

    # Instance 1: write data
    store1 = VectorStore(persist_directory=db_path)
    chunk = make_sample_chunk(chunk_id="persisted_chunk")
    store1.upsert_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]])
    assert store1.collection.count() == 1

    # Instance 2: reopen same directory
    store2 = VectorStore(persist_directory=db_path)
    assert store2.collection.count() == 1
    assert store2.has_document("doc1") is True
