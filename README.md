# AI Knowledge Base Chatbot

A modular, evidence-grounded Retrieval-Augmented Generation (RAG) assistant built with Streamlit, ChromaDB, Sentence Transformers, and the Google Gemini API using the modern `google-genai` SDK.

---

## Overview

The **AI Knowledge Base Chatbot** delivers reliable, evidence-backed answers strictly grounded in uploaded documents (PDF, TXT, and Markdown). It employs structure-aware chunking, local sentence embeddings, persistent vector indexing, prompt-injection safeguards, and Google Gemini to generate answers accompanied by verifiable, structured source citations.

When documents do not contain sufficient evidence to answer a question, the chatbot explicitly refuses to answer rather than hallucinating or speculating.

---

## Features

- **Multi-Format Document Ingestion**: In-memory parsing for `.pdf`, `.txt`, `.md`, and `.markdown` files.
- **Page-Aware PDF Extraction**: PyMuPDF extracts text page-by-page, preserving 1-based page numbers for citation attribution.
- **Structure-Aware Chunking**: Splits text into 500-word target chunks with 100-word overlap, respecting paragraph and sentence boundaries while keeping headings and list items attached to their following text.
- **Page Boundary Preservation**: Chunks from different PDF pages are strictly kept separate to prevent false multi-page citations.
- **Local Dense Embeddings**: Generates 384-dimensional vector embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` with zero external API dependencies for vectorization.
- **Persistent Vector Database**: Stores vectors and chunk metadata locally in ChromaDB (`academic_knowledge_base` collection) with idempotent upserts and document-level deletion.
- **Untrusted Context Isolation**: Retrieved document text is fenced within explicit untrusted evidence boundaries, instructing Gemini to treat it purely as passive reference data and ignore any embedded override instructions.
- **Verifiable Source Citations**: Every grounded answer includes structured citation cards detailing document filename, page attribution, chunk ID, and verbatim text excerpts.
- **Evidence-Based Refusal Policy**: Distinguishes between an empty knowledge base and missing topic evidence, returning clear, honest notices when information is absent.
- **Configurable System Prompt**: System role, tone, format, permitted topics, and rules are defined in `config/system_prompt.json`, editable via the Streamlit administrator interface with live hot-reloading.
- **Scope & Out-of-Scope Interception**: Intercepts blatant off-topic requests (e.g. recipes, creative writing, horoscopes) before making LLM calls, saving API costs and tokens.
- **Secret & Credential Masking**: API keys are masked in application logs and error messages; the chatbot functions in an evidence-inspection mode even when an API key is not configured.

---

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Runtime** | Python 3.11 / 3.12 | Core application language |
| **Web Interface** | Streamlit | Interactive web UI with Chat, Ingestion, Search, and Admin tabs |
| **Document Processing** | PyMuPDF (`pymupdf`) | PDF page-by-page text extraction and validation |
| **Vector Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | Local 384-dimensional dense vector embeddings |
| **Vector Database** | ChromaDB | Local persistent vector storage with cosine similarity |
| **Large Language Model** | Google Gemini (`gemini-2.5-flash`) | Context-grounded response generation via `google-genai` SDK |
| **Configuration** | `python-dotenv` & JSON | Environment management and dynamic prompt configuration |
| **Testing & Quality** | `pytest` | Comprehensive unit and integration test suite (82 tests) |

---

## Architecture & Workflow

```
[ User Uploads (PDF / TXT / MD) ]
               │
               ▼
   [ Document Processor ] ──> In-memory text extraction & page attribution
               │
               ▼
      [ Chunker Module ] ──> Structure-aware chunking (500 words / 100 overlap)
               │
               ▼
   [ Embedding Service ] ──> Local all-MiniLM-L6-v2 embedding generation
               │
               ▼
     [ ChromaDB Store ] ──> Persistent vector collection (academic_knowledge_base)
               │
═══════════════╪═══════════════════════════════════════════════════════════════
               ▼
[ User Question in Chat UI ]
               │
               ▼
   [ Scope Sanity Check ] ──> Intercepts blatant off-topic queries early
               │
               ▼
    [ Semantic Retriever ] ──> Top-k nearest-neighbor vector search
               │
               ▼
    [ Prompt Assembler ] ──> Fences untrusted evidence + system instructions
               │
               ▼
    [ Google Gemini LLM ] ──> Grounded generation (temperature=0.2)
               │
               ▼
  [ Answer & Citations ] ──> Displays markdown answer with verifiable source cards
```

---

## Project Structure

```
AI KNOWLEDGE BASED CHATBOT/
├── app.py                     # Streamlit application entrypoint (4 UI tabs)
├── requirements.txt           # Python package dependencies
├── .env.example               # Template for required environment variables
├── .gitignore                 # Git ignore configuration
├── README.md                  # Project documentation
│
├── config/
│   └── system_prompt.json     # Dynamic system prompt, role, tone, and permitted topics
│
├── data/
│   ├── .gitkeep
│   └── sample_documents/      # Reference documents for testing and demonstrations
│       ├── prompt_injection_guide.md
│       └── database_indexing_principles.md
│
├── chroma_db/                 # Local persistent ChromaDB storage (git-ignored)
│   └── .gitkeep
│
├── src/                       # Core application source modules
│   ├── __init__.py
│   ├── document_processor.py  # Text extraction and document validation
│   ├── chunker.py             # Structure-aware chunking and page boundary isolation
│   ├── embeddings.py          # Sentence-transformers embedding wrapper
│   ├── vector_store.py        # ChromaDB client and collection management
│   ├── retriever.py           # Semantic search and relevance ranking
│   ├── prompt_builder.py      # System prompt configuration and prompt injection fences
│   ├── rag.py                 # RAG orchestration, Gemini API integration, and citations
│   ├── chatbot.py             # High-level chatbot interface
│   └── llm_service.py         # LLM service helpers
│
├── scripts/                   # Verification, diagnostic, and evaluation scripts
│   ├── verify_setup.py        # Environment and dependency verification
│   ├── run_evaluation.py      # 5-scenario evaluation runner (mocked & live modes)
│   ├── test_gemini_live.py    # Targeted Gemini live integration test
│   ├── diagnose_and_test.py   # Diagnostic pipeline tracing across test cases
│   └── diagnose_rag.py        # Vector store inspection tool
│
└── tests/                     # Automated test suite (82 passing tests)
    ├── __init__.py
    ├── test_setup.py          # Setup and configuration tests
    ├── test_extraction.py     # Document parser and extraction tests
    ├── test_chunking.py       # Chunking boundaries, overlap, and metadata tests
    ├── test_embeddings.py     # Embedding service tests
    ├── test_vector_store.py   # ChromaDB upsert, query, and deletion tests
    ├── test_retriever.py      # Retrieval ranking and score tests
    ├── test_prompt_config.py  # Prompt configuration and boundary tests
    ├── test_rag.py            # RAG pipeline, citations, and refusal tests
    ├── test_evaluation.py     # Automated evaluation of the 5 core scenarios
    └── test_integration.py    # End-to-end integration and Streamlit import tests
```

---

## Installation & Configuration

### Prerequisites
- **Python**: Version 3.11 or 3.12 installed on your system.
- **Git**: For version control.

### 1. Create a Virtual Environment

**Windows PowerShell:**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to create your local `.env` file:

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Edit `.env` and provide your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GEMINI_MODEL=gemini-2.5-flash
```

> **Security Note**: Never commit `.env` or expose API keys. The file `.env` is included in `.gitignore`.

---

## Running the Application

Start the Streamlit application:

```powershell
streamlit run app.py
```

Or using the virtual environment interpreter directly:
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Once started, open your browser and navigate to:
```
http://localhost:8501
```

---

## Document Workflow: Upload, Process, Index & Chat

The application separates extraction from vector indexing to give you complete visibility into generated chunks before writing to disk:

1. **Upload Documents**:
   - Go to the **📁 Document Ingestion & Indexing** tab.
   - Drag and drop or browse for `.pdf`, `.txt`, `.md`, or `.markdown` files.
   - Example documents are available in `data/sample_documents/`.
2. **Process Uploaded Files**:
   - Click **🚀 Process Uploaded Files**.
   - Text is extracted, cleaned, and split into structure-aware chunks in memory.
   - Review extracted chunks, page assignments, and token estimates in the status table.
3. **Index into ChromaDB**:
   - Click **⚡ Index Processed Documents**.
   - The local Sentence Transformer generates 384-dimensional dense vectors and commits them to the persistent ChromaDB collection.
4. **Ask Questions**:
   - Switch to the **💬 RAG Chatbot** tab.
   - Type questions grounded in your uploaded documents.
5. **Review Answers & Citations**:
   - Grounded answers cite sources inline (e.g. `[Doc: guide.pdf, Page: 2]`).
   - Expand the **📚 Source Citations** accordion below each response to review the supporting document name, page number, chunk ID, and exact verbatim excerpt.
   - If the indexed documents do not contain the answer, the chatbot explicitly states that the information cannot be found, rather than guessing.

---

## Administrator Prompt & Boundaries Configuration

Administrators can customize chatbot behavior without code modifications using the **⚙️ Prompt & Boundaries** tab:

- **Role & Persona**: e.g., *"Document-Grounded Academic Assistant"*.
- **Tone & Style**: e.g., *"objective, professional, concise, and helpful"*.
- **Answer Format**: Formatting guidelines for answers and citations.
- **Permitted Topics**: List of authorized academic topics. Blatant off-topic queries outside these domains are intercepted early to preserve API quota.
- **System Instructions**: Strict evidence-grounding constraints, conflicting information rules, and safety boundaries.

Clicking **💾 Save & Apply Settings** validates and saves updates to `config/system_prompt.json`. The chatbot hot-reloads these settings on every query without requiring an application restart.

---

## Testing & Verification

### 1. Verify Setup & Dependencies
Check your Python version, package imports, directory structure, and configuration integrity:
```powershell
python scripts/verify_setup.py
```

### 2. Run the Full Automated Test Suite (93 tests)
Run all unit and integration tests (deterministic, offline, requires no API key):
```powershell
pytest -v
```

### 3. Run the Five Core RAG Evaluation Scenarios
The evaluation suite benchmarks the five fundamental RAG operational scenarios:

```powershell
python scripts/run_evaluation.py
```

| Scenario # | Operational Scenario | Verification Criteria | Deterministic Mode |
| :---: | :--- | :--- | :---: |
| **1** | **Supported Knowledge Question** | Grounded answer factually consistent with text; valid source filename and page citation. | **PASSED ✅** |
| **2** | **Missing Information Refusal** | Question seeks facts absent from knowledge base; explicit refusal without fabrication or API waste. | **PASSED ✅** |
| **3** | **Prompt Injection Defense** | User adversarial overrides ignored; untrusted document content isolated in fenced blocks. | **PASSED ✅** |
| **4** | **Out-of-Scope Query Boundary** | Blatant non-academic queries (e.g., cooking recipes) intercepted prior to LLM call. | **PASSED ✅** |
| **5** | **Conflicting Information** | Contradictory facts across documents identified; both sources cited without fabricated resolution. | **PASSED ✅** |

> **Evaluation Notes**:
> - All 5 scenarios pass deterministically in mocked/unit test suites (`pytest -v` reports **93 passed**).
> - Live Gemini verification is supported when `GEMINI_API_KEY` is configured. Targeted real Gemini requests for supported, missing, and prompt injection queries have been verified against live API responses.
> - High-frequency live batch evaluation runs may encounter Google AI Studio free-tier rate limits (`429 RESOURCE_EXHAUSTED`). In such cases, use the offline deterministic evaluation runner.


---

## Safe Re-Indexing Procedure

The persistent ChromaDB collection is stored in `./chroma_db` and preserved between runs:

1. To update chunks or re-index documents after changing chunking settings, open the **📁 Document Ingestion & Indexing** tab.
2. Click **🗑️ Delete Document Chunks** next to any indexed document to remove its vectors safely from ChromaDB.
3. Re-process and re-index the document. Other documents in the collection remain unaffected.

---

## Limitations

- **No OCR Support**: Scanned image-only PDFs lacking a text layer cannot be parsed; PyMuPDF requires an embedded digital text layer.
- **Text Layer Only**: Non-text elements such as diagrams, images, and visual charts within PDFs are not parsed as visual inputs.
- **Prompt Injection Defense Scope**: Untrusted context fencing significantly mitigates standard prompt overrides and jailbreak attempts. However, prompt-level fencing alone does not provide a mathematical guarantee against all advanced indirect injection vectors. High-security environments should implement multi-layered defenses.
- **Conversational Memory**: Chat history is bounded to recent dialogue turns to prevent conversational context from overwhelming retrieved document evidence.

---

## License

This project is developed for educational and academic research purposes under the MIT License.
