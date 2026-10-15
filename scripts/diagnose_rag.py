"""Diagnostic script to inspect ChromaDB persistent collection and trace query pipeline."""
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv

load_dotenv()

from src.vector_store import VectorStore
from src.retriever import Retriever
from src.rag import RAGService
from src.prompt_builder import check_query_topic_scope

def diagnose():
    print("=== 1. Vector Store Inspection ===")
    store = VectorStore()
    stats = store.get_stats()
    print(f"Collection stats: {stats}")
    
    collection = store.collection
    data = collection.get(include=["metadatas", "documents"])
    ids = data.get("ids", [])
    print(f"Total chunk IDs found: {len(ids)}")
    
    unique_sources = set()
    if ids:
        for m in data.get("metadatas", []):
            if m and "source_file" in m:
                unique_sources.add(m["source_file"])
        print(f"Indexed sources ({len(unique_sources)}): {list(unique_sources)}")
        print("\nSample Chunks (up to 3):")
        for i in range(min(3, len(ids))):
            meta = data["metadatas"][i]
            doc = data["documents"][i][:120].replace("\n", " ")
            print(f"  [{i+1}] ID: {ids[i]} | Source: {meta.get('source_file')} | Page: {meta.get('page_number')} | Text: {doc}...")
    else:
        print("WARNING: VectorStore is completely empty! No documents have been indexed yet.")

    print("\n=== 2. Pipeline Tracing for: 'What is prompt injection?' ===")
    query = "What is prompt injection?"
    
    # Check scope
    is_in_scope, scope_reason = check_query_topic_scope(query)
    print(f"Scope check: in_scope={is_in_scope}, reason='{scope_reason}'")
    
    # Retrieval
    retriever = Retriever(vector_store=store)
    retrieved = retriever.retrieve(query=query, top_k=5)
    print(f"Retrieved chunks count: {len(retrieved)}")
    for idx, c in enumerate(retrieved):
        dist = f"{c.distance:.4f}" if c.distance is not None else "None"
        sim = f"{c.similarity_score:.4f}" if c.similarity_score is not None else "None"
        print(f"  [{idx+1}] File: {c.source_file} | Page: {c.page_number} | Dist: {dist} | Sim: {sim}")
        print(f"        Text preview: {c.text[:150].replace(chr(10), ' ')}...")
        
    print("\n=== 3. RAG Service Execution ===")
    rag = RAGService(retriever=retriever)
    response = rag.answer_question(query)
    print(f"RAG response has_sufficient_evidence: {response.has_sufficient_evidence}")
    print(f"RAG citations count: {len(response.citations)}")
    print(f"RAG answer:\n{response.answer}")

if __name__ == "__main__":
    diagnose()
