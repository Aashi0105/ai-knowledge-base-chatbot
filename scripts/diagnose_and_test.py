"""Comprehensive diagnostic and test script for RAG answer quality and prompt injection."""
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
load_dotenv()

from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag import RAGService
from src.prompt_builder import check_query_topic_scope, load_prompt_config
from src.document_processor import process_document
from src.chunker import chunk_document
from src.embeddings import get_embedding_service


def run_pipeline_trace():
    print("=" * 60)
    print("STEP 1: INSPECT INITIAL PERSISTENT CHROMADB STATE")
    print("=" * 60)
    store = VectorStore()
    stats = store.get_stats()
    print(f"Collection: {stats['collection_name']}")
    print(f"Persist dir: {stats['persist_directory']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Document IDs: {stats['document_ids']}")

    prompt_config = load_prompt_config()
    permitted_topics = prompt_config.permitted_topics
    print(f"Permitted topics: {permitted_topics}")

    print("\n" + "=" * 60)
    print("STEP 2: TRACE PIPELINE FOR 'What is prompt injection?' (Initial State)")
    print("=" * 60)
    query = "What is prompt injection?"

    # Check topic scope
    is_in_scope, refusal_reason = check_query_topic_scope(query, permitted_topics)
    print(f"Topic scope check: in_scope={is_in_scope}, refusal_reason={refusal_reason}")

    # Check retrieval
    retriever = Retriever(vector_store=store)
    chunks = retriever.retrieve(query=query, top_k=4)
    print(f"Retrieved chunks count: {len(chunks)}")
    for idx, c in enumerate(chunks, 1):
        print(f"  [{idx}] source={c.source}, page={c.page}, dist={c.distance}, sim={c.similarity_score}")
        print(f"      text={c.text[:100]}...")

    # Run RAG
    rag = RAGService(retriever=retriever)
    res = rag.answer_question(query)
    print(f"RAG Response has_sufficient_evidence: {res.has_sufficient_evidence}")
    print(f"RAG Response answer:\n{res.answer}")
    print(f"RAG Citations: {res.citations}")

    print("\n" + "=" * 60)
    print("STEP 3: TEST CASE A - INDEX A NON-PROMPT-INJECTION DOCUMENT")
    print("=" * 60)
    # Sample Document: Database indexing principles (B-Tree)
    db_doc_text = (
        "# Database Storage Engines and B-Tree Indexes\n\n"
        "A B-Tree is a self-balancing search tree data structure that maintains sorted data "
        "and allows searches, sequential access, insertions, and deletions in logarithmic time.\n\n"
        "B-Trees are optimized for systems that read and write large blocks of memory, such as "
        "hard disk drives and solid-state disks. The depth of a B-Tree is typically very low, "
        "making disk accesses minimal during indexing lookups.\n"
    )
    doc_res = process_document(db_doc_text.encode("utf-8"), "database_indexes.md")
    chunks = chunk_document(doc_res.segments, chunk_size=200, chunk_overlap=30)
    emb_service = get_embedding_service()
    embeddings = emb_service.embed_texts([c.text for c in chunks])
    store.upsert_chunks(chunks, embeddings)
    print(f"Indexed 'database_indexes.md': {len(chunks)} chunks.")

    # Test A1: Ask a question explicitly answered in the document
    q_supported = "What is a B-Tree and what operations does it allow in logarithmic time?"
    print(f"\nSub-test A1: Supported Question: '{q_supported}'")
    res_a1 = rag.answer_question(q_supported)
    print(f"Answer:\n{res_a1.answer}")
    print(f"Citations ({len(res_a1.citations)}):")
    for c in res_a1.citations:
        print(f"  - [{c.source}, Page {c.page}]: {c.excerpt[:80]}...")

    # Test A2 / Case B: Ask 'What is prompt injection?' when knowledge base only has database document
    print(f"\nSub-test A2 / Case B: Unsupported Question: '{query}'")
    chunks_unsupported = retriever.retrieve(query=query, top_k=4)
    print(f"Retrieved {len(chunks_unsupported)} chunks from DB document:")
    for idx, c in enumerate(chunks_unsupported, 1):
        print(f"  [{idx}] source={c.source}, page={c.page}, dist={c.distance}, sim={c.similarity_score}")
        print(f"      text={c.text[:100]}...")

    res_b = rag.answer_question(query)
    print(f"Answer:\n{res_b.answer}")
    print(f"Citations count: {len(res_b.citations)}")
    print(f"has_sufficient_evidence: {res_b.has_sufficient_evidence}")

    print("\n" + "=" * 60)
    print("STEP 4: TEST CASE C - INDEX A PROMPT INJECTION EXPLAINER DOCUMENT")
    print("=" * 60)
    pi_doc_text = (
        "# Security in Large Language Models: Prompt Injection\n\n"
        "Prompt injection is a security vulnerability where an attacker crafts malicious input "
        "that manipulates an AI or Large Language Model into ignoring its original instructions, "
        "system prompt, or guardrails, and executing unintended commands instead.\n\n"
        "There are two primary categories of prompt injection:\n"
        "1. Direct Prompt Injection (also called Jailbreaking): The user directly instructs the model "
        "to bypass safety filters, override system rules, or adopt an unrestricted persona.\n"
        "2. Indirect Prompt Injection: Malicious instructions are embedded inside external data sources, "
        "such as uploaded documents, web pages, or database records, which the LLM processes and "
        "mistakenly treats as instructions rather than untrusted data.\n\n"
        "Mitigation strategies include strict separation of instructions from untrusted data, "
        "input validation, and robust privilege isolation.\n"
    )
    pi_doc_res = process_document(pi_doc_text.encode("utf-8"), "prompt_injection_overview.md")
    pi_chunks = chunk_document(pi_doc_res.segments, chunk_size=200, chunk_overlap=30)
    pi_embeddings = emb_service.embed_texts([c.text for c in pi_chunks])
    store.upsert_chunks(pi_chunks, pi_embeddings)
    print(f"Indexed 'prompt_injection_overview.md': {len(pi_chunks)} chunks.")

    print(f"\nSub-test C: Re-ask '{query}' with relevant document indexed")
    chunks_c = retriever.retrieve(query=query, top_k=4)
    print(f"Retrieved {len(chunks_c)} chunks:")
    for idx, c in enumerate(chunks_c, 1):
        print(f"  [{idx}] source={c.source}, page={c.page}, dist={c.distance}, sim={c.similarity_score}")
        print(f"      text={c.text[:100]}...")

    res_c = rag.answer_question(query)
    print(f"Answer:\n{res_c.answer}")
    print(f"Citations ({len(res_c.citations)}):")
    for c in res_c.citations:
        print(f"  - [{c.source}, Page {c.page}]: {c.excerpt[:80]}...")


if __name__ == "__main__":
    run_pipeline_trace()
