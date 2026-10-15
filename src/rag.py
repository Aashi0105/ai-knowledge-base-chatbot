"""RAG (Retrieval-Augmented Generation) pipeline for AI Knowledge Based Chatbot.

Orchestrates semantic chunk retrieval, prompt assembly with untrusted document
evidence isolation, Gemini API generation, and structured citation compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Optional, Sequence
from dotenv import load_dotenv

from src.prompt_builder import (
    DEFAULT_SYSTEM_PROMPT_PATH,
    GREETING_RESPONSE,
    SystemPromptConfig,
    build_rag_prompt,
    check_query_topic_scope,
    is_conversational_greeting,
    load_prompt_config,
    load_system_instruction,
)
from src.retriever import RetrievedChunk, Retriever

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_RELEVANCE_THRESHOLD = 0.30  # Minimum cosine similarity cutoff for relevant evidence


@dataclass(frozen=True)
class Citation:
    """Represents a structured source citation grounded in retrieved document chunks.

    Attributes:
        source: Filename of the source document.
        page: 1-based page number for PDFs, or None for TXT/Markdown.
        excerpt: Short verbatim text excerpt from the supporting chunk.
        chunk_id: Unique identifier of the supporting chunk.
    """
    source: str
    page: Optional[int]
    excerpt: str
    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "excerpt": self.excerpt,
            "chunk_id": self.chunk_id,
        }


@dataclass
class RAGResponse:
    """Response payload containing generated answer and source citations."""
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    has_sufficient_evidence: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "has_sufficient_evidence": self.has_sufficient_evidence,
            "error": self.error,
        }


def make_citations_from_chunks(chunks: Sequence[RetrievedChunk], max_excerpt_len: int = 180) -> list[Citation]:
    """Generate structured citations from actual retrieved chunks."""
    citations: list[Citation] = []
    for chunk in chunks:
        excerpt = chunk.text.strip()
        if len(excerpt) > max_excerpt_len:
            excerpt = excerpt[:max_excerpt_len].rstrip() + "..."
        citations.append(
            Citation(
                source=chunk.source,
                page=chunk.page,
                excerpt=excerpt,
                chunk_id=chunk.chunk_id,
            )
        )
    return citations


def format_gemini_error(err: Exception) -> str:
    """Format Gemini API exceptions into actionable, user-friendly messages while masking secrets."""
    err_str = str(err)
    err_lower = err_str.lower()

    # 1. Quota & Rate Limits (429 RESOURCE_EXHAUSTED)
    if "429" in err_str or "resource_exhausted" in err_lower or "quota" in err_lower:
        return (
            "Unable to generate response from Gemini API: Quota Exceeded (429 Resource Exhausted). "
            "The request limit on your Gemini API key has been reached. Please check your Google AI Studio "
            "quota and billing details, or try again later. (Retrieved source evidence is preserved in citations below)."
        )

    # 2. Authentication & Invalid Credentials (400 / 401 / 403 API_KEY_INVALID)
    if any(k in err_lower for k in ("api_key_invalid", "invalid api key", "unauthenticated", "permission_denied", "api key not valid")):
        return (
            "Unable to generate response from Gemini API: Authentication Failed (Invalid API Key). "
            "Please check that your GEMINI_API_KEY in the .env file is active and valid."
        )

    # 3. Model Not Found or Unavailable (404 NOT_FOUND)
    if any(k in err_lower for k in ("not_found", "not found", "is not supported", "model not found")):
        return (
            "Unable to generate response from Gemini API: Model Unavailable (404 Not Found). "
            "The configured model is not accessible with your API key. Please check GEMINI_MODEL in your .env file."
        )

    # 4. Network / Connection Errors
    if any(k in err_lower for k in ("connection", "timeout", "network", "wsarecv", "connection closed", "failed to connect")):
        return (
            "Unable to generate response from Gemini API: Network Connection Error. "
            "A network interruption or timeout occurred while communicating with Gemini services. Please check your internet connection."
        )

    # 5. Generic Fallback
    clean_err = type(err).__name__
    return (
        f"Unable to generate response from Gemini API: {clean_err}. "
        "Please verify your API key, model configuration, and network connection."
    )


class RAGService:
    """End-to-end RAG orchestrator coordinating retrieval, prompt assembly, and Gemini."""

    INSUFFICIENT_EVIDENCE_RESPONSE = (
        "I could not find sufficient information in the indexed knowledge base documents "
        "to answer your question. Please verify that relevant documents covering this topic are uploaded and indexed."
    )
    EMPTY_KNOWLEDGE_BASE_RESPONSE = (
        "I could not find sufficient information to answer your question because the knowledge base "
        "is currently empty (0 documents indexed). Please upload and index documents in the "
        "'Document Ingestion & Indexing' tab before asking questions."
    )

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        gemini_client: Optional[Any] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        config_path: Optional[Path | str] = None,
    ) -> None:
        """Initialize RAGService.

        Args:
            retriever: Retriever instance to query ChromaDB.
            gemini_client: Optional pre-configured or mocked Gemini client.
            model_name: Gemini model name. Defaults to GEMINI_MODEL or 'gemini-2.5-flash'.
            api_key: Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            config_path: Path to system prompt configuration JSON.
        """
        self.retriever = retriever or Retriever()
        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or DEFAULT_GEMINI_MODEL
        )
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self.config_path = config_path or DEFAULT_SYSTEM_PROMPT_PATH
        self._gemini_client = gemini_client

    @property
    def gemini_client(self) -> Any:
        """Lazily initialize Google GenAI client."""
        if self._gemini_client is None:
            if not self.api_key or not self.api_key.strip():
                raise ValueError("GEMINI_API_KEY is not configured in environment or .env file.")
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize google-genai client: {e}") from e
        return self._gemini_client

    def answer_question(
        self,
        question: str,
        top_k: int = 4,
        conversation_history: Optional[Sequence[dict[str, str]]] = None,
        score_threshold: Optional[float] = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> RAGResponse:
        """Answer a user question grounded in retrieved knowledge base chunks.

        Enforces system prompt boundaries, topic constraints, prompt-injection defense,
        and evidence verification.

        Args:
            question: User's natural-language question.
            top_k: Maximum number of chunks to retrieve for evidence.
            conversation_history: Optional recent dialogue turns for context.
            score_threshold: Optional similarity threshold for chunk relevance. Defaults to DEFAULT_RELEVANCE_THRESHOLD (0.30).

        Returns:
            RAGResponse containing answer, citations, chunks, and status indicators.
        """
        if not question or not question.strip():
            return RAGResponse(
                answer="Please enter a valid question.",
                citations=[],
                retrieved_chunks=[],
                has_sufficient_evidence=False,
            )

        # 0. Intercept conversational greetings cleanly (e.g. "hey", "hello", "hi there")
        if is_conversational_greeting(question):
            return RAGResponse(
                answer=GREETING_RESPONSE,
                citations=[],
                retrieved_chunks=[],
                has_sufficient_evidence=True,
                error=None,
            )


        # 1. Load active system prompt configuration from single source of truth
        try:
            prompt_config = load_prompt_config(self.config_path)
            effective_system_instruction = prompt_config.build_effective_system_instruction()
            permitted_topics = prompt_config.permitted_topics
        except Exception:
            # Fallback to safe defaults if config file is temporarily inaccessible
            effective_system_instruction = load_system_instruction(self.config_path)
            permitted_topics = ["Academic knowledge base topics"]

        # 2. Topic boundary check: intercept blatant off-topic non-academic queries
        is_in_scope, refusal_reason = check_query_topic_scope(question, permitted_topics)
        if not is_in_scope and refusal_reason:
            return RAGResponse(
                answer=refusal_reason,
                citations=[],
                retrieved_chunks=[],
                has_sufficient_evidence=False,
            )

        # 3. Retrieve relevant chunks using the semantic retriever
        try:
            chunks = self.retriever.retrieve(
                query=question.strip(),
                top_k=top_k,
                score_threshold=score_threshold,
            )
        except Exception as e:
            return RAGResponse(
                answer=f"An error occurred while retrieving knowledge base documents: {e}",
                citations=[],
                retrieved_chunks=[],
                has_sufficient_evidence=False,
                error=str(e),
            )

        # 4. Check for empty or insufficient evidence
        if not chunks:
            # Check if knowledge base is completely empty
            vstore = getattr(self.retriever, "vector_store", None)
            if vstore is not None:
                try:
                    stats = vstore.get_stats()
                    if stats.get("total_chunks", 0) == 0:
                        return RAGResponse(
                            answer=self.EMPTY_KNOWLEDGE_BASE_RESPONSE,
                            citations=[],
                            retrieved_chunks=[],
                            has_sufficient_evidence=False,
                        )
                except Exception:
                    pass

            return RAGResponse(
                answer=self.INSUFFICIENT_EVIDENCE_RESPONSE,
                citations=[],
                retrieved_chunks=[],
                has_sufficient_evidence=False,
            )

        # 5. Create structured citations from the verified retrieved chunks
        citations = make_citations_from_chunks(chunks)

        # 6. Check for API key availability before attempting LLM call
        if not self.api_key or not self.api_key.strip():
            return RAGResponse(
                answer=(
                    "Gemini API key is not configured. Please set `GEMINI_API_KEY` in your `.env` "
                    "file to enable conversational answer generation. Retrieved relevant source "
                    "evidence is shown in the citations below."
                ),
                citations=citations,
                retrieved_chunks=chunks,
                has_sufficient_evidence=True,
                error="GEMINI_API_KEY_NOT_CONFIGURED",
            )

        # 7. Build prompt with untrusted evidence isolation and call Gemini API
        prompt = build_rag_prompt(
            question=question,
            evidence_chunks=chunks,
            conversation_history=conversation_history,
        )

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=effective_system_instruction,
                temperature=0.2,  # Low temperature for factual, grounded answers
            )

            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            answer_text = response.text.strip() if (response and response.text) else ""
            if not answer_text:
                answer_text = self.INSUFFICIENT_EVIDENCE_RESPONSE

            # Detect if model explicitly refused due to missing evidence or unsupported query
            refusal_indicators = (
                "could not find information regarding this in the provided knowledge base",
                "do not contain sufficient information to answer this question",
                "does not contain sufficient information to answer this question",
                "not mentioned in the provided",
                "not mentioned in the uploaded",
                "not mentioned in the retrieved",
                "provided documents do not contain",
                "provided context does not contain",
                "provided documents do not mention",
                "could not find sufficient information in the indexed knowledge base documents",
            )
            is_refusal = any(phrase in answer_text.lower() for phrase in refusal_indicators)

            return RAGResponse(
                answer=answer_text,
                citations=[] if is_refusal else citations,
                retrieved_chunks=chunks,
                has_sufficient_evidence=not is_refusal,
                error=None,
            )

        except Exception as e:
            # Mask API keys and protect secrets from leaking in error messages
            err_msg = str(e)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[MASKED_KEY]")

            friendly_answer = format_gemini_error(e)
            return RAGResponse(
                answer=friendly_answer,
                citations=citations,
                retrieved_chunks=chunks,
                has_sufficient_evidence=True,
                error=err_msg,
            )

