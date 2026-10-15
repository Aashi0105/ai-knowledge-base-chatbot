"""Google Gemini LLM service integration module using modern google-genai SDK.

Handles client initialization, configuration management, and grounded inference.
"""

from src.rag import DEFAULT_GEMINI_MODEL, RAGService

__all__ = ["DEFAULT_GEMINI_MODEL", "RAGService"]
