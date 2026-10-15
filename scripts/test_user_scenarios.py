import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.rag import RAGService

service = RAGService()

print("=" * 60)
print("TEST 1: Greeting ('hey')")
print("=" * 60)
r1 = service.answer_question("hey")
print("Answer:", r1.answer)
print("Citations count:", len(r1.citations))
print("Retrieved chunks count:", len(r1.retrieved_chunks))
print("Has sufficient evidence:", r1.has_sufficient_evidence)

print("\n" + "=" * 60)
print("TEST 2: Absent topic ('what is the project threadtheory about?')")
print("=" * 60)
r2 = service.answer_question("what is the project threadtheory about?")
print("Answer:", r2.answer)
print("Citations count:", len(r2.citations))
print("Retrieved chunks count:", len(r2.retrieved_chunks))
print("Has sufficient evidence:", r2.has_sufficient_evidence)

print("\n" + "=" * 60)
print("TEST 3: Relevant topic ('What is a B-Tree?')")
print("=" * 60)
r3 = service.answer_question("What is a B-Tree?")
print("Answer:", r3.answer[:140], "...")
print("Citations count:", len(r3.citations))
if r3.citations:
    print("Citations sources:", [c.source for c in r3.citations])
print("Retrieved chunks count:", len(r3.retrieved_chunks))
print("Error (if any):", r3.error[:80] if r3.error else None)
