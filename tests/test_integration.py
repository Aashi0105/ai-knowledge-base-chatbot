"""Integration tests for document processing, chunking, ChromaDB retrieval, and RAG answering (Phase 4)."""

import importlib
from pathlib import Path
import pymupdf
from src.chunker import chunk_document
from src.document_processor import process_document
from src.embeddings import EmbeddingService
from src.rag import RAGService
from src.retriever import Retriever
from src.vector_store import VectorStore
from tests.test_embeddings import MockTransformerModel
from tests.test_rag import MockGeminiClient


def create_sample_pdf() -> bytes:
    """Helper to generate a multi-page PDF locally."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((50, 72), "AI Knowledge Base Chatbot integration test Page 1 content with several words.")
    p2 = doc.new_page()
    p2.insert_text((50, 72), "Second page content for verifying end-to-end extraction and chunking.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# Extraction and chunking work together for a sample document (Phase 2 regression).
def test_end_to_end_extraction_and_chunking():
    pdf_bytes = create_sample_pdf()
    doc_result = process_document(pdf_bytes, "integration_test.pdf")

    assert doc_result.status == "Processed"
    assert len(doc_result.segments) == 2
    assert doc_result.error is None

    chunks = chunk_document(doc_result.segments, chunk_size=5, chunk_overlap=1)
    assert len(chunks) > 0

    # Ensure page attribution is maintained through the full pipeline
    pages_in_chunks = {c.page for c in chunks}
    assert pages_in_chunks == {1, 2}

    for c in chunks:
        assert c.document_id == doc_result.document_id
        assert c.source == "integration_test.pdf"
        assert len(c.text.strip()) > 0


# Phase 2 chunk can be indexed and retrieved end to end using mocked embeddings (Phase 3 regression).
def test_phase2_chunks_indexed_and_retrieved_end_to_end(tmp_path: Path):
    pdf_bytes = create_sample_pdf()
    doc_result = process_document(pdf_bytes, "cloud_computing.pdf")
    assert doc_result.status == "Processed"

    chunks = chunk_document(doc_result.segments, chunk_size=10, chunk_overlap=2)
    assert len(chunks) > 0

    mock_model = MockTransformerModel(dim=12)
    embedding_service = EmbeddingService(model=mock_model)
    vector_store = VectorStore(persist_directory=str(tmp_path))
    retriever = Retriever(vector_store=vector_store, embedding_service=embedding_service)

    vectors = embedding_service.embed_texts([c.text for c in chunks])
    indexed_count = vector_store.upsert_chunks(chunks, vectors)
    assert indexed_count == len(chunks)

    retrieved = retriever.retrieve("integration test", top_k=2)
    assert len(retrieved) > 0
    assert retrieved[0].source == "cloud_computing.pdf"
    assert retrieved[0].document_id == doc_result.document_id
    assert retrieved[0].page in {1, 2}


# Full end-to-end test: Ingest -> Chunk -> Index -> RAG Question Answering with Citations (Phase 4).
def test_full_rag_pipeline_end_to_end(tmp_path: Path):
    pdf_bytes = create_sample_pdf()
    doc_result = process_document(pdf_bytes, "operating_systems_manual.pdf")
    chunks = chunk_document(doc_result.segments, chunk_size=8, chunk_overlap=2)

    mock_model = MockTransformerModel(dim=8)
    embedding_service = EmbeddingService(model=mock_model)
    vector_store = VectorStore(persist_directory=str(tmp_path))
    retriever = Retriever(vector_store=vector_store, embedding_service=embedding_service)

    vectors = embedding_service.embed_texts([c.text for c in chunks])
    vector_store.upsert_chunks(chunks, vectors)

    mock_gemini = MockGeminiClient(
        "Page 1 discusses the AI Knowledge Base Chatbot integration test, as supported by [operating_systems_manual.pdf, Page 1]."
    )
    rag_service = RAGService(
        retriever=retriever,
        gemini_client=mock_gemini,
        api_key="valid_mock_key",
    )

    response = rag_service.answer_question("What does Page 1 discuss?")

    assert response.has_sufficient_evidence is True
    assert "Page 1" in response.answer
    assert len(response.citations) > 0
    assert response.citations[0].source == "operating_systems_manual.pdf"
    assert response.citations[0].page in {1, 2}
    assert response.citations[0].excerpt != ""


# Streamlit application imports without error.
def test_streamlit_app_imports_cleanly():
    app_module = importlib.import_module("app")
    assert hasattr(app_module, "main")
    assert hasattr(app_module, "process_uploaded_files")
    assert hasattr(app_module, "index_documents")
    assert hasattr(app_module, "get_rag_service")
