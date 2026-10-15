"""AI Knowledge Base Chatbot - Streamlit Application Entrypoint.

Phase 5: Configurable System Prompt, Chatbot Boundaries, and Grounded RAG Chat.
Combines document ingestion, chunking, persistent ChromaDB indexing, semantic
retrieval, administrator prompt configuration, and Gemini-powered question answering.
"""

from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv

from src.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_document
from src.document_processor import generate_document_id, process_document
from src.embeddings import get_embedding_service
from src.prompt_builder import (
    ConfigurationError,
    SystemPromptConfig,
    load_prompt_config,
    save_prompt_config,
)
from src.rag import DEFAULT_GEMINI_MODEL, RAGService
from src.retriever import Retriever
from src.vector_store import VectorStore

load_dotenv()


@st.cache_resource
def get_vector_store() -> VectorStore:
    """Return a cached VectorStore instance."""
    return VectorStore()


@st.cache_resource
def get_retriever() -> Retriever:
    """Return a cached Retriever instance wired to the VectorStore and EmbeddingService."""
    store = get_vector_store()
    service = get_embedding_service()
    return Retriever(vector_store=store, embedding_service=service)


@st.cache_resource
def get_rag_service() -> RAGService:
    """Return a cached RAGService instance."""
    retriever = get_retriever()
    return RAGService(retriever=retriever)


def init_session_state() -> None:
    """Initialize session state keys for documents, search, and chat history."""
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = {}  # doc_id -> doc_dict
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role", "content", "citations"}


def handle_remove_document(doc_id: str) -> None:
    """Remove a document from session state and delete its chunks from ChromaDB."""
    store = get_vector_store()
    try:
        store.delete_document(doc_id)
    except Exception as e:
        st.error(f"Error removing chunks from ChromaDB for document '{doc_id}': {e}")

    if doc_id in st.session_state.processed_docs:
        del st.session_state.processed_docs[doc_id]


def process_uploaded_files(
    uploaded_files: list,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Process uploaded files for extraction and chunking (Phase 2 pipeline)."""
    store = get_vector_store()

    for uploaded_file in uploaded_files:
        content = uploaded_file.getvalue()
        filename = uploaded_file.name
        doc_id = generate_document_id(filename, content)

        if doc_id in st.session_state.processed_docs:
            continue

        doc_result = process_document(content, filename)

        if doc_result.status == "Processed" and doc_result.segments:
            chunks = chunk_document(
                doc_result.segments,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if chunks:
                final_status = "Processed"
                err_msg = None
            else:
                final_status = "Failed"
                err_msg = "No non-empty chunks could be generated from document text."
        else:
            chunks = []
            final_status = "Failed"
            err_msg = doc_result.error or "Document extraction produced no readable text."

        is_already_indexed = store.has_document(doc_id)
        indexing_status = "Indexed" if is_already_indexed else "Not indexed"
        indexed_count = len(chunks) if is_already_indexed else 0

        st.session_state.processed_docs[doc_id] = {
            "document_id": doc_id,
            "filename": filename,
            "file_type": doc_result.file_type,
            "status": final_status,
            "indexing_status": indexing_status,
            "indexed_chunks_count": indexed_count,
            "segments": doc_result.segments,
            "chunks": chunks,
            "warnings": doc_result.warnings,
            "error": err_msg,
            "indexing_error": None,
        }


def index_documents(doc_ids: list[str]) -> None:
    """Index processed chunks into ChromaDB with local sentence embeddings."""
    store = get_vector_store()
    service = get_embedding_service()

    for doc_id in doc_ids:
        doc = st.session_state.processed_docs.get(doc_id)
        if not doc or doc["status"] != "Processed" or not doc["chunks"]:
            continue

        doc["indexing_status"] = "Indexing"
        try:
            chunks = doc["chunks"]
            chunk_texts = [c.text for c in chunks]
            embeddings = service.embed_texts(chunk_texts)
            indexed_count = store.upsert_chunks(chunks, embeddings)

            doc["indexing_status"] = "Indexed"
            doc["indexed_chunks_count"] = indexed_count
            doc["indexing_error"] = None
        except Exception as e:
            doc["indexing_status"] = "Indexing failed"
            doc["indexing_error"] = str(e)


def main() -> None:
    st.set_page_config(
        page_title="AI Knowledge Base Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    store = get_vector_store()
    retriever = get_retriever()
    rag_service = get_rag_service()

    st.title("AI Knowledge Base Chatbot")
    st.subheader("Document-Based Question Answering using RAG")

    st.markdown(
        "**Phase 5: Configurable System Prompt, Chatbot Boundaries & Grounded RAG Chat**  \n"
        "Ask questions grounded strictly in uploaded academic documents. Chatbot behavior, role, tone, "
        "permitted topics, and safety boundaries are dynamically configured from `config/system_prompt.json`."
    )

    st.markdown("---")

    # --- Sidebar Configuration & Index Statistics ---
    with st.sidebar:
        st.header("⚙️ System Status")

        # ChromaDB Persistent Statistics
        stats = store.get_stats()
        st.markdown("### 🗄️ Knowledge Base Index")
        st.write(f"• **Collection**: `{stats['collection_name']}`")
        st.write(f"• **Indexed Documents**: `{stats['total_documents']}`")
        st.write(f"• **Indexed Chunks**: `{stats['total_chunks']}`")

        # Gemini API Status Check
        has_api_key = bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY").strip())
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        st.markdown("### 🤖 Gemini LLM")
        if has_api_key:
            st.success("🟢 API Key Configured")
        else:
            st.warning("🟡 API Key Missing (Evidence retrieval works)")
        st.caption(f"Model: `{model_name}`")

        st.markdown("---")
        st.subheader("RAG Parameters")
        rag_top_k = st.slider("Retrieved Context Chunks (k)", min_value=1, max_value=8, value=4)
        rag_threshold = st.slider(
            "Relevance Cutoff (Similarity)",
            min_value=0.0,
            max_value=0.80,
            value=0.30,
            step=0.05,
            help="Minimum cosine similarity required for a chunk to be passed as evidence to Gemini.",
        )

        st.markdown("---")
        st.subheader("Ingestion Settings")
        chunk_size = st.number_input(
            "Chunk Size (approx. words)",
            min_value=50,
            max_value=2000,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (approx. words)",
            min_value=0,
            max_value=chunk_size - 1,
            value=min(DEFAULT_CHUNK_OVERLAP, chunk_size - 1),
            step=10,
        )

        st.markdown("---")
        if st.session_state.chat_history:
            if st.button("🧹 Clear Chat History", use_container_width=True):
                st.session_state.chat_history.clear()
                st.rerun()

        if st.session_state.processed_docs:
            if st.button("🗑️ Clear Session Documents", use_container_width=True):
                st.session_state.processed_docs.clear()
                st.session_state.search_results.clear()
                st.rerun()

    # --- Main Navigation Tabs ---
    tab_chat, tab_ingest, tab_search, tab_admin = st.tabs(
        [
            "💬 RAG Chatbot",
            "📁 Document Ingestion & Indexing",
            "🔍 Semantic Search & Chunks",
            "⚙️ Prompt & Boundaries",
        ]
    )

    # =========================================================================
    # TAB 1: RAG Chatbot
    # =========================================================================
    with tab_chat:
        st.subheader("💬 Ask Questions About Your Knowledge Base")

        if stats["total_chunks"] == 0:
            st.info(
                "💡 The knowledge base is currently empty. "
                "Please upload and index documents in the **Document Ingestion & Indexing** tab to begin."
            )

        # Render conversation history
        for msg in st.session_state.chat_history:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])

                # Render source citations if available
                citations = msg.get("citations", [])
                if citations:
                    with st.expander(f"📚 Source Citations ({len(citations)})", expanded=False):
                        for c_idx, cit in enumerate(citations, start=1):
                            page_label = f"Page {cit['page']}" if cit["page"] is not None else "Plain Text / Markdown"
                            st.markdown(f"**Citation [{c_idx}]**: 📄 `{cit['source']}` — *{page_label}*")
                            st.caption(f"Chunk ID: `{cit['chunk_id']}`")
                            st.markdown(f"> *\"{cit['excerpt']}\"*")
                            st.markdown("---")

        # Chat Input
        user_question = st.chat_input("Type your question based on uploaded documents...")
        if user_question:
            # 1. Display user prompt immediately
            st.chat_message("user").markdown(user_question)
            st.session_state.chat_history.append({"role": "user", "content": user_question, "citations": []})

            # 2. Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching document context and generating grounded answer..."):
                    response = rag_service.answer_question(
                        question=user_question,
                        top_k=int(rag_top_k),
                        conversation_history=st.session_state.chat_history[:-1],
                        score_threshold=float(rag_threshold),
                    )

                st.markdown(response.answer)

                citations_dicts = [c.to_dict() for c in response.citations]
                if citations_dicts:
                    with st.expander(f"📚 Source Citations ({len(citations_dicts)})", expanded=True):
                        for c_idx, cit in enumerate(citations_dicts, start=1):
                            page_label = f"Page {cit['page']}" if cit["page"] is not None else "Plain Text / Markdown"
                            st.markdown(f"**Citation [{c_idx}]**: 📄 `{cit['source']}` — *{page_label}*")
                            st.caption(f"Chunk ID: `{cit['chunk_id']}`")
                            st.markdown(f"> *\"{cit['excerpt']}\"*")
                            st.markdown("---")
                elif not response.has_sufficient_evidence:
                    current_stats = store.get_stats()
                    if current_stats["total_chunks"] == 0:
                        st.info(
                            "💡 **Knowledge Base is Empty**: No documents are currently indexed in ChromaDB. "
                            "Upload and click **'⚡ Index Processed Documents'** in the **Document Ingestion & Indexing** tab."
                        )
                    else:
                        st.info(
                            f"💡 **Information Not in Knowledge Base**: The indexed documents ({current_stats['total_documents']} doc(s), "
                            f"{current_stats['total_chunks']} chunk(s)) do not contain sufficient evidence to answer this query. "
                            "Upload documents covering this topic to expand the knowledge base."
                        )

            # 3. Save assistant message and citations in session state
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "citations": citations_dicts,
                    "has_sufficient_evidence": response.has_sufficient_evidence,
                }
            )

    # =========================================================================
    # TAB 2: Document Ingestion & Indexing
    # =========================================================================
    with tab_ingest:
        st.subheader("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or Markdown documents",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            help="Files are processed directly in-memory.",
        )

        col_b1, col_b2 = st.columns([3, 7])
        with col_b1:
            process_clicked = st.button("🚀 Process Uploaded Files", type="primary", use_container_width=True)

        if process_clicked and uploaded_files:
            with st.spinner("Extracting text and chunking uploaded documents..."):
                process_uploaded_files(uploaded_files, int(chunk_size), int(chunk_overlap))
            st.success("Document extraction and chunking complete!")

        docs = list(st.session_state.processed_docs.values())

        if docs:
            st.markdown("---")
            st.subheader("2. Document Processing & Indexing Status")

            unindexed_doc_ids = [
                d["document_id"]
                for d in docs
                if d["status"] == "Processed" and d["indexing_status"] != "Indexed"
            ]

            if unindexed_doc_ids:
                st.warning(
                    f"⚠️ **Action Required**: {len(unindexed_doc_ids)} document(s) have been processed into chunks "
                    "but are **not yet indexed** into ChromaDB. Click **'⚡ Index Processed Documents'** below to make them searchable!"
                )

            c_idx_btn, c_idx_info = st.columns([3, 7])
            with c_idx_btn:
                if st.button(
                    f"⚡ Index Processed Documents ({len(unindexed_doc_ids)})",
                    disabled=len(unindexed_doc_ids) == 0,
                    use_container_width=True,
                ):
                    with st.spinner("Generating embeddings and writing to ChromaDB..."):
                        index_documents(unindexed_doc_ids)
                    st.success("ChromaDB vector indexing complete!")
                    st.rerun()

            with c_idx_info:
                st.caption("Generates 384-dimensional vector embeddings locally and persists in ChromaDB.")

            # Summary Metrics
            m1, m2, m3, m4 = st.columns(4)
            total_docs = len(docs)
            processed_count = sum(1 for d in docs if d["status"] == "Processed")
            indexed_count = sum(1 for d in docs if d["indexing_status"] == "Indexed")
            total_chunks = sum(len(d["chunks"]) for d in docs if d["status"] == "Processed")

            m1.metric("Uploaded Documents", total_docs)
            m2.metric("Extracted (Processed)", processed_count)
            m3.metric("Vector Indexed", indexed_count)
            m4.metric("Total Chunks", total_chunks)

            # Document Inventory
            st.markdown("### Document Inventory")
            for doc in docs:
                doc_id = doc["document_id"]
                p_status = doc["status"]
                i_status = doc["indexing_status"]

                p_icon = "✅" if p_status == "Processed" else "❌"
                if i_status == "Indexed":
                    i_badge = "🟢 Indexed"
                elif i_status == "Indexing failed":
                    i_badge = "🔴 Indexing failed"
                elif i_status == "Indexing":
                    i_badge = "🟡 Indexing..."
                else:
                    i_badge = "⚪ Not indexed"

                with st.container(border=True):
                    c_info, c_stats, c_action = st.columns([5, 4, 2])

                    with c_info:
                        st.markdown(f"**{p_icon} {doc['filename']}** (`{doc['file_type']}`)")
                        st.caption(f"Document ID: `{doc_id}`")
                        if doc["error"]:
                            st.error(f"Extraction Error: {doc['error']}")
                        if doc["indexing_error"]:
                            st.error(f"Indexing Error: {doc['indexing_error']}")
                        for warning in doc["warnings"]:
                            st.warning(f"Warning: {warning}")

                    with c_stats:
                        st.write(
                            f"• **Processing Status**: `{p_status}`  \n"
                            f"• **Indexing Status**: `{i_badge}`  \n"
                            f"• **Chunks in Vector Store**: {doc['indexed_chunks_count']} / {len(doc['chunks'])}"
                        )

                    with c_action:
                        if p_status == "Processed" and i_status != "Indexed":
                            if st.button("Index", key=f"idx_{doc_id}", help="Index this document into ChromaDB"):
                                with st.spinner(f"Indexing {doc['filename']}..."):
                                    index_documents([doc_id])
                                st.rerun()

                        if st.button("Delete", key=f"del_{doc_id}", help="Remove from session and delete from ChromaDB"):
                            handle_remove_document(doc_id)
                            st.rerun()

                    if p_status == "Processed":
                        with st.expander(f"🔍 Inspect Document Chunks: {doc['filename']}", expanded=False):
                            tab_chunks, tab_segments = st.tabs(
                                [f"Chunks ({len(doc['chunks'])})", f"Segments ({len(doc['segments'])})"]
                            )
                            with tab_chunks:
                                for chunk in doc["chunks"]:
                                    page_str = f"Page {chunk.page}" if chunk.page is not None else "Full Document"
                                    st.markdown(
                                        f"**Chunk #{chunk.chunk_index + 1}** (`{chunk.chunk_id}`) — *{page_str}* "
                                        f"({len(chunk.text.split())} words)"
                                    )
                                    st.text_area(
                                        label=f"Content ({chunk.chunk_id})",
                                        value=chunk.text,
                                        height=100,
                                        key=f"chunk_view_{chunk.chunk_id}",
                                        disabled=True,
                                        label_visibility="collapsed",
                                    )
                            with tab_segments:
                                for idx, seg in enumerate(doc["segments"]):
                                    page_str = f"Page {seg.page}" if seg.page is not None else f"Segment {idx + 1}"
                                    st.markdown(f"**{page_str}** — ({len(seg.text.split())} words)")
                                    st.text_area(
                                        label=f"Segment ({doc_id}_{idx})",
                                        value=seg.text,
                                        height=120,
                                        key=f"seg_view_{doc_id}_{idx}",
                                        disabled=True,
                                        label_visibility="collapsed",
                                    )
        else:
            st.info("No documents uploaded yet. Upload files above to begin.")

    # =========================================================================
    # TAB 3: Semantic Search & Direct Chunk Inspection
    # =========================================================================
    with tab_search:
        st.subheader("🔍 Direct Semantic Search & Retrieval Inspection")

        if stats["total_chunks"] == 0:
            st.warning("The vector database is currently empty. Please upload and index documents in Tab 2 first.")
        else:
            col_q, col_k, col_thresh = st.columns([6, 2, 2])
            with col_q:
                query_text = st.text_input(
                    "Search query:",
                    value=st.session_state.last_query,
                    placeholder="e.g., What are the key properties of database transactions (ACID)?",
                )
            with col_k:
                top_k = st.slider("Top Results (k)", min_value=1, max_value=10, value=4, key="direct_search_k")
            with col_thresh:
                search_thresh = st.slider(
                    "Min Similarity",
                    min_value=0.0,
                    max_value=0.80,
                    value=0.0,
                    step=0.05,
                    key="direct_search_thresh",
                    help="Filter out weak matches below this similarity threshold.",
                )

            col_sbtn, _ = st.columns([2, 8])
            with col_sbtn:
                search_clicked = st.button("Search Chunks", type="primary", use_container_width=True)

            if search_clicked and query_text.strip():
                st.session_state.last_query = query_text.strip()
                with st.spinner("Embedding query and retrieving nearest chunks..."):
                    try:
                        results = retriever.retrieve(
                            query=query_text.strip(),
                            top_k=int(top_k),
                            score_threshold=float(search_thresh) if search_thresh > 0.0 else None,
                        )
                        st.session_state.search_results = results
                    except Exception as e:
                        st.error(f"Search retrieval error: {e}")
                        st.session_state.search_results = []

            if st.session_state.search_results:
                st.markdown(f"### Retrieved Chunks for: *\"{st.session_state.last_query}\"*")
                for rank, chunk in enumerate(st.session_state.search_results, start=1):
                    page_badge = f"Page {chunk.page}" if chunk.page is not None else "Text/Markdown (No Page)"
                    sim = chunk.similarity_score
                    if sim >= 0.60:
                        relevance_badge = "🟢 **High Relevance**"
                    elif sim >= 0.30:
                        relevance_badge = "🟡 **Moderate Relevance**"
                    else:
                        relevance_badge = "🔴 **Low Relevance (Weak Match - Below Cutoff)**"

                    with st.container(border=True):
                        st.markdown(
                            f"**Rank #{rank}** | {relevance_badge} | 📄 Source: `{chunk.source}` ({page_badge}) | "
                            f"Similarity: `{sim:.4f}` | Distance: `{chunk.distance:.4f}`"
                        )
                        st.caption(f"Chunk ID: `{chunk.chunk_id}` | Document ID: `{chunk.document_id}`")
                        st.markdown(chunk.text)
            elif search_clicked and not st.session_state.search_results:
                st.info("No matching chunks found above the selected similarity threshold.")


    # =========================================================================
    # TAB 4: Administrator: Prompt & Boundaries Configuration
    # =========================================================================
    with tab_admin:
        st.subheader("⚙️ Administrator: System Prompt & Chatbot Boundaries")
        st.caption(
            "This section configures the single source of truth in `config/system_prompt.json`. "
            "Applied changes dynamically build the effective system instruction for Gemini calls "
            "and enforce topic boundaries."
        )

        try:
            current_config = load_prompt_config()
        except Exception as e:
            st.error(f"Failed to load active system prompt configuration: {e}")
            current_config = SystemPromptConfig(
                role="Document-Grounded AI Knowledge Assistant",
                tone="objective, professional, concise, and helpful",
                answer_format="Markdown response with verifiable source citations",
                permitted_topics=["Academic knowledge base topics"],
                system_prompt="Answer strictly from retrieved document evidence.",
            )

        with st.form("admin_prompt_form"):
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                admin_role = st.text_input(
                    "Chatbot Role",
                    value=current_config.role,
                    help="Identifies the role and persona of the assistant.",
                )
            with col_r2:
                admin_tone = st.text_input(
                    "Response Tone & Style",
                    value=current_config.tone,
                    help="Specifies tone (e.g. objective, professional, concise).",
                )

            admin_answer_format = st.text_area(
                "Answer Format & Citation Rules",
                value=current_config.answer_format,
                height=90,
                help="Defines structured format requirements and citation formatting guidelines.",
            )

            admin_permitted_topics = st.text_area(
                "Permitted Topics (one per line)",
                value="\n".join(current_config.permitted_topics),
                height=110,
                help="Topics the assistant is permitted to address.",
            )

            admin_system_prompt = st.text_area(
                "Core Behavioral Instructions & Guardrails",
                value=current_config.system_prompt,
                height=220,
                help="Core guidelines for factual grounding, refusal rules, and boundary enforcement.",
            )

            save_submitted = st.form_submit_button("💾 Save and Apply Configuration", type="primary")

        if save_submitted:
            cleaned_topics = [t.strip() for t in admin_permitted_topics.split("\n") if t.strip()]
            new_candidate = SystemPromptConfig(
                role=admin_role.strip(),
                tone=admin_tone.strip(),
                answer_format=admin_answer_format.strip(),
                permitted_topics=cleaned_topics,
                system_prompt=admin_system_prompt.strip(),
            )

            try:
                save_prompt_config(new_candidate)
                st.success("✅ Configuration saved to `config/system_prompt.json` and applied to subsequent chat requests!")
                st.rerun()
            except ConfigurationError as ce:
                st.error(f"Validation Error: {ce}")
            except Exception as ex:
                st.error(f"Error persisting configuration: {ex}")

        # Live Preview of Effective System Instruction
        with st.expander("👁️ Preview Effective System Instruction (as sent to Gemini)", expanded=False):
            st.code(current_config.build_effective_system_instruction(), language="markdown")


if __name__ == "__main__":
    main()
