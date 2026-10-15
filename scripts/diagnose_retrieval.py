import sys
import os

sys.path.insert(0, os.path.abspath("."))
from src.retriever import Retriever

retriever = Retriever()
queries = [
    "what is the project threadtheory about?",
    "hey",
    "What is a B-Tree?",
    "What is prompt injection?",
]

for q in queries:
    results = retriever.retrieve(q, top_k=4)
    print(f"\n=== Query: '{q}' ===")
    for r in results:
        print(f"  source={r.source} dist={r.distance:.4f} sim={r.similarity_score:.4f} text_preview={r.text[:60]}...")
