"""Live verification script for real Google Gemini API and end-to-end RAG pipeline.

Verifies:
1. GEMINI_API_KEY presence in .env (without logging or exposing the secret).
2. Minimal direct Gemini generation request using configured model.
3. Grounded RAG question-answering with a small sample document and citation verification.
4. Insufficient evidence handling for unsupported questions.
5. Error handling and secret masking on API failure.

Uses an isolated temporary vector database directory to ensure the persistent chroma_db
collection remains completely untouched.
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Ensure stdout uses UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.chunker import chunk_document
from src.document_processor import process_document
from src.embeddings import EmbeddingService
from src.rag import DEFAULT_GEMINI_MODEL, RAGService
from src.retriever import Retriever
from src.vector_store import VectorStore


def run_live_verification() -> int:
    print("=" * 65)
    print("Real Gemini Integration & End-to-End RAG Verification")
    print("=" * 65)

    # 1. Inspect API Key Presence
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        print("[FAIL] GEMINI_API_KEY is not configured in .env.")
        print("Please configure GEMINI_API_KEY in your local .env file.")
        return 2

    # Do not print, display, or expose the key!
    masked_preview = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    print(f"[OK] GEMINI_API_KEY found in .env (Format: {masked_preview}, length: {len(api_key)} chars).")

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    print(f"[OK] Target Gemini Model: {model_name}")

    # 2. Minimal Real Gemini API Test
    print("\n--- Test 1: Direct Minimal Gemini API Call ---")
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        direct_resp = client.models.generate_content(
            model=model_name,
            contents="Respond strictly with the single word: VERIFIED",
        )
        resp_text = direct_resp.text.strip() if direct_resp and direct_resp.text else ""
        print(f"[OK] Direct Gemini API call succeeded. Response: {resp_text[:60]}")
    except Exception as e:
        err_msg = str(e)
        if api_key in err_msg:
            err_msg = err_msg.replace(api_key, "[MASKED_KEY]")
        print(f"[FAIL] Direct Gemini API call failed: {type(e).__name__}: {err_msg}")
        return 1

    # 3. End-to-End RAG Test with Isolated Temporary Vector Store
    print("\n--- Test 2: End-to-End Grounded RAG with Sample Document ---")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        sample_filename = "quantum_sort_research.txt"
        sample_content = (
            "Title: The Quantum Sorting Algorithm\n"
            "Author: Dr. Elena Vance\n"
            "Affiliation: Cambridge Quantum Computing Laboratory\n"
            "Date: October 2024\n\n"
            "Abstract:\n"
            "The Quantum Sorting Algorithm achieves an asymptotic running time of O(N log log N) "
            "on fault-tolerant quantum computers. It requires a minimum coherence time of 45 milliseconds "
            "and uses topological qubit gates to maintain quantum superposition during comparison operations.\n\n"
            "Implementation Requirements:\n"
            "1. Coherence time strictly exceeding 45 milliseconds.\n"
            "2. Cryogenic cooling down to 15 millikelvin.\n"
            "3. Topological error-correcting codes."
        ).encode("utf-8")

        # Process document
        doc_result = process_document(sample_content, sample_filename)
        assert doc_result.status == "Processed", f"Document processing failed: {doc_result.error}"

        # Chunk with structure-aware chunker
        chunks = chunk_document(doc_result.segments)
        assert len(chunks) > 0, "No chunks generated"

        # Embed and index into temporary vector store
        embed_service = EmbeddingService()
        temp_store = VectorStore(persist_directory=tmp_dir)
        vectors = embed_service.embed_texts([c.text for c in chunks])
        temp_store.upsert_chunks(chunks, vectors)

        retriever = Retriever(vector_store=temp_store, embedding_service=embed_service)
        rag_service = RAGService(
            retriever=retriever,
            model_name=model_name,
            api_key=api_key,
        )

        # Question A: Supported Question
        supported_q = "What is the asymptotic running time of the Quantum Sorting Algorithm and who is the author?"
        print(f"\nAsking Supported Question: '{supported_q}'")
        res_supported = rag_service.answer_question(supported_q)

        print(f"Answer received:\n{res_supported.answer}")
        print(f"Citations count: {len(res_supported.citations)}")
        for cit in res_supported.citations:
            print(f"  • Source: {cit.source}, Page: {cit.page}, Excerpt: {cit.excerpt[:80]}...")

        if res_supported.error:
            print(f"[FAIL] Supported query returned error: {res_supported.error}")
            return 1
        if not res_supported.has_sufficient_evidence:
            print("[FAIL] Expected sufficient evidence for supported query.")
            return 1
        if not res_supported.citations or res_supported.citations[0].source != sample_filename:
            print("[FAIL] Missing or invalid source citation.")
            return 1

        # Check grounding in the answer text
        ans_lower = res_supported.answer.lower()
        if "elena vance" not in ans_lower and "elena" not in ans_lower:
            print("[FAIL] Expected author 'Elena Vance' in answer.")
            return 1
        if "log" not in ans_lower and "o(n" not in ans_lower:
            print("[FAIL] Expected asymptotic complexity in answer.")
            return 1

        print("[OK] Supported question answered correctly with verifiable citation.")

        # Question B: Unsupported Question (Fact not in document)
        unsupported_q = "What are the undergraduate admission grade requirements for Cambridge University?"
        print(f"\nAsking Unsupported Question: '{unsupported_q}'")
        res_unsupported = rag_service.answer_question(unsupported_q)

        print(f"Answer received:\n{res_unsupported.answer}")
        # The answer must clearly indicate missing/insufficient evidence or refusal
        unsupp_ans = res_unsupported.answer.lower()
        is_refusal = (
            "not find information" in unsupp_ans
            or "not find sufficient" in unsupp_ans
            or "could not find" in unsupp_ans
            or "do not contain" in unsupp_ans
            or "does not contain" in unsupp_ans
            or "not mentioned" in unsupp_ans
            or "not provided" in unsupp_ans
            or "no information" in unsupp_ans
            or not res_unsupported.has_sufficient_evidence
        )
        if not is_refusal:
            print("[FAIL] Chatbot did not follow insufficient-evidence refusal behavior.")
            return 1

        print("[OK] Unsupported question handled with proper evidence refusal.")

        # 4. Error Handling Verification
        print("\n--- Test 3: API Failure & Secret Masking ---")
        bad_key = "AIzaSyFakeKeyForTestingErrorHandling999"
        bad_service = RAGService(
            retriever=retriever,
            model_name=model_name,
            api_key=bad_key,
        )
        bad_res = bad_service.answer_question(supported_q)
        if bad_key in str(bad_res.error) or bad_key in str(bad_res.answer):
            print("[FAIL] Secret key was leaked in error message or answer!")
            return 1
        print(f"[OK] Error handled safely without exposing credentials (Error: {str(bad_res.error)[:70]}...).")

    print("\n" + "=" * 65)
    print("ALL REAL GEMINI INTEGRATION CHECKS PASSED SUCCESSFULLY ✅")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(run_live_verification())
