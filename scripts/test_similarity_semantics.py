import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.retriever import Retriever

retriever = Retriever()
test_queries = [
    # Supported queries
    ("What is a B-Tree?", "Supported DB"),
    ("Explain B-Tree search and logarithmic time operations", "Supported DB"),
    ("What is prompt injection in large language models?", "Supported PI"),
    ("Explain direct versus indirect prompt injection jailbreaking", "Supported PI"),
    # Marginal / weakly related queries
    ("What are the advantages of tree data structures in computer science?", "Marginal DB"),
    ("How do security vulnerabilities affect modern artificial intelligence?", "Marginal PI"),
    # Completely unsupported / unrelated queries
    ("what is the project threadtheory about?", "Unsupported ThreadTheory"),
    ("Who invented the steam engine in the industrial revolution?", "Unsupported History"),
    ("What is the recipe for chocolate cake?", "Unsupported Recipe"),
    ("hey", "Greeting"),
    ("hello there", "Greeting"),
]

print(f"{'Query':<60} | {'Category':<22} | {'Top Source':<28} | {'Dist':<6} | {'Sim':<6}")
print("-" * 135)
for q, cat in test_queries:
    results = retriever.retrieve(q, top_k=2)
    if results:
        top = results[0]
        print(f"{q:<60} | {cat:<22} | {top.source:<28} | {top.distance:.4f} | {top.similarity_score:.4f}")
    else:
        print(f"{q:<60} | {cat:<22} | {'None':<28} | {'N/A':<6} | {'N/A':<6}")
