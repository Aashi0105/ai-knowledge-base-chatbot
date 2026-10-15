"""Main RAG Chatbot orchestration module for AI Knowledge Based Chatbot.

Coordinates end-to-end question answering across retrieval, prompt synthesis,
Gemini response generation, structured citation compilation, and system prompt configuration.
"""

from src.prompt_builder import (
    ConfigurationError,
    SystemPromptConfig,
    load_prompt_config,
    save_prompt_config,
)
from src.rag import Citation, RAGResponse, RAGService

__all__ = [
    "Citation",
    "ConfigurationError",
    "RAGResponse",
    "RAGService",
    "SystemPromptConfig",
    "load_prompt_config",
    "save_prompt_config",
]
