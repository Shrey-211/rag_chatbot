# Project Structure

This document explains the organization of the RAG Chatbot project to help you navigate the codebase.

## 📁 Directory Overview

```
rag_chatbot/
├── api/                    # FastAPI Backend Application
├── src/                    # Core Source Code
├── webapp/                 # React Frontend Application
├── tests/                  # Test Suite
├── scripts/                # Utility Scripts
├── docs/                   # Documentation
├── data/                   # Data Storage (gitignored)
├── .github/                # GitHub Configuration
└── [Root Files]            # Configuration and Documentation
```

## 🔍 Detailed Structure

### `/api` - Backend API

**Purpose:** FastAPI application that exposes REST endpoints for document indexing and querying.

```
api/
├── __init__.py             # Package initialization
├── main.py                 # FastAPI app, endpoints, lifecycle management
├── models.py               # Pydantic models for request/response
└── database.py             # SQLite database for document metadata
```

**Key Files:**
- `main.py`: Core application with endpoints like `/query`, `/index/file`, `/documents`
- `models.py`: Data validation models (QueryRequest, IndexResponse, etc.)
- `database.py`: Document metadata storage and retrieval

### `/src` - Core Source Code

**Purpose:** Modular components implementing RAG functionality.

```
src/
├── adapters/               # Pluggable adapters (Strategy pattern)
│   ├── llm/               # Large Language Model adapters
│   │   ├── base.py        # Abstract LLM interface
│   │   ├── ollama.py      # Ollama local LLM
│   │   ├── openai.py      # OpenAI API
│   │   └── mock.py        # Mock LLM for testing
│   ├── embedding/         # Text embedding adapters
│   │   ├── base.py        # Abstract embedding interface
│   │   ├── local.py       # sentence-transformers (local)
│   │   └── openai.py      # OpenAI embeddings
│   └── vision/            # Vision model adapters
│       ├── base.py        # Abstract vision interface
│       └── ollama.py      # Ollama vision models (LLaVA, etc.)
│
├── extractors/            # Document text extraction
│   ├── base.py           # Abstract extractor + factory
│   ├── pdf.py            # PDF extraction (PyPDF2)
│   ├── docx.py           # Word document extraction
│   ├── txt.py            # Plain text files
│   ├── image.py          # Image OCR (Tesseract)
│   └── table.py          # CSV/JSON table extraction
│
├── vectorstore/          # Vector database implementations
│   ├── base.py          # Abstract vector store interface
│   ├── chroma.py        # ChromaDB implementation
│   ├── faiss.py         # FAISS implementation
│   └── memory.py        # In-memory (for testing)
│
├── retriever/           # Document retrieval logic
│   └── retriever.py     # Query, retrieve, rank documents
│
├── services/            # Business logic services
│   └── personal_info_extractor.py  # Extract personal info from documents
│
├── config/              # Configuration management
│   └── config.py        # Load config from YAML/env vars
│
└── utils/               # Utility functions
    ├── chunking.py      # Text chunking algorithms
    ├── prompts.py       # Prompt templates
    └── dependency_checker.py  # Check OCR dependencies
```

**Architecture Patterns:**

1. **Strategy Pattern:** Adapters allow swapping implementations (e.g., Ollama ↔ OpenAI)
2. **Factory Pattern:** ExtractorFactory creates appropriate extractor for file type
3. **Dependency Injection:** Components receive dependencies through constructors
4. **Interface Segregation:** Abstract base classes define clear contracts

### `/webapp` - Frontend Application

**Purpose:** React + TypeScript web interface for interacting with the RAG system.

```
webapp/
├── src/
│   ├── App.tsx           # Main application component
│   ├── main.tsx          # Entry point
│   ├── App.css           # Styles
│   └── vite-env.d.ts     # TypeScript definitions
├── public/               # Static assets
├── index.html            # HTML template
├── package.json          # Node dependencies
├── vite.config.ts        # Vite configuration
├── tsconfig.json         # TypeScript configuration
└── Dockerfile            # Frontend Docker image

```

**Tech Stack:**
- **React 18:** UI library
- **TypeScript:** Type safety
- **Vite:** Fast build tool
- **Fetch API:** REST API calls

### `/tests` - Test Suite

**Purpose:** Comprehensive tests for all components.

```
tests/
├── __init__.py
├── test_e2e.py                    # End-to-end API tests
├── test_llm_adapters.py           # LLM adapter tests
├── test_embedding_adapters.py     # Embedding adapter tests
├── test_extractors.py             # Document extractor tests
├── test_image_extractor.py        # OCR tests
└── test_vectorstore.py            # Vector store tests
```

**Testing Strategy:**
- **Unit Tests:** Test individual components in isolation
- **Integration Tests:** Test component interactions
- **E2E Tests:** Test full API workflows
- **Mocking:** Use mocks for external dependencies

**Run Tests:**
```bash
pytest tests/ -v                    # All tests
pytest tests/test_extractors.py    # Specific file
pytest --cov=src                    # With coverage
```

### `/scripts` - Utility Scripts

**Purpose:** CLI tools for common tasks.

```
scripts/
├── index_documents.py           # Index documents from directory
├── query.py                     # Query from command line
├── quickstart.sh                # Setup script (Unix)
├── quickstart.bat               # Setup script (Windows)
├── setup_windows_ocr.ps1        # OCR setup (Windows)
└── setup_ocr_manual.md          # Manual OCR setup guide
```

**Usage Examples:**
```bash
# Index all files in a directory
python scripts/index_documents.py ./documents/

# Query interactively
python scripts/query.py --interactive

# Query with sources
python scripts/query.py "What is RAG?" --show-sources
```

### `/docs` - Documentation

**Purpose:** Project documentation and guides.

```
docs/
├── architecture.md                    # System architecture
├── switching_providers.md             # How to switch LLM/embeddings
├── embedding_guide.md                 # Embedding models guide
├── image_processing_setup.md          # OCR setup guide
├── personal_info_extraction_guide.md  # Personal info extraction
└── PROJECT_STRUCTURE.md               # This file
```

### `/data` - Data Storage (gitignored)

**Purpose:** Runtime data storage.

```
data/
├── chroma/              # ChromaDB vector store
│   ├── chroma.sqlite3  # ChromaDB metadata
│   └── [embeddings]    # Vector embeddings
├── faiss/              # FAISS index files (if using FAISS)
├── uploads/            # Uploaded document files
├── sample/             # Sample documents for testing
└── documents.db        # SQLite document metadata
```

**Note:** This directory is gitignored and created at runtime.

### `/.github` - GitHub Configuration

**Purpose:** GitHub-specific files for better open source experience.

```
.github/
├── ISSUE_TEMPLATE/
│   ├── bug_report.md       # Bug report template
│   ├── feature_request.md  # Feature request template
│   └── documentation.md    # Documentation issue template
├── PULL_REQUEST_TEMPLATE.md  # PR template
└── FUNDING.yml              # Sponsorship info (optional)
```

### Root Configuration Files

```
├── LICENSE                  # MIT License
├── README.md               # Main project documentation
├── SETUP_GUIDE.md          # Beginner-friendly setup guide
├── CONTRIBUTING.md         # Contribution guidelines
├── CODE_OF_CONDUCT.md      # Community standards
├── SECURITY.md             # Security policy
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Python project metadata
├── config.yaml             # Main configuration file
├── config.example.yaml     # Example configuration
├── Dockerfile              # Backend Docker image
├── docker-compose.yml      # Multi-container setup
├── Makefile                # Common development tasks
└── .gitignore             # Git ignore rules
```

## 🔄 Data Flow

### 1. Document Indexing Flow

```
User uploads file
    ↓
API endpoint (/index/file)
    ↓
ExtractorFactory selects extractor
    ↓
Extractor extracts text (with OCR if needed)
    ↓
Text is chunked (chunking.py)
    ↓
Chunks are embedded (EmbeddingAdapter)
    ↓
Embeddings stored in VectorStore
    ↓
Metadata saved to SQLite database
    ↓
Success response returned
```

### 2. Query Flow

```
User asks question
    ↓
API endpoint (/query)
    ↓
Query is embedded (EmbeddingAdapter)
    ↓
Retriever searches VectorStore
    ↓
Top-K relevant chunks retrieved
    ↓
Chunks formatted as context
    ↓
Prompt built with context + query
    ↓
LLM generates answer (LLMAdapter)
    ↓
Answer + sources returned to user
```

### 3. Personal Info Extraction Flow (Optional)

```
User uploads document
    ↓
After text extraction and indexing
    ↓
VisionAdapter analyzes document pages
    ↓
Structured personal info extracted
    ↓
Entities saved to database
    ↓
Retrieved during queries for enriched context
```

## 🧩 Key Design Patterns

### 1. Adapter Pattern

**Purpose:** Allow different implementations to be swapped easily.

**Example:**
```python
# All LLMs implement the same interface
class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        pass

# Can swap between implementations
llm = OllamaAdapter()  # or OpenAIAdapter()
response = llm.generate(prompt)
```

### 2. Factory Pattern

**Purpose:** Create objects without specifying exact class.

**Example:**
```python
# Factory selects correct extractor based on file type
extractor = ExtractorFactory.create("document.pdf")
content = extractor.extract("document.pdf")
```

### 3. Dependency Injection

**Purpose:** Components receive dependencies, making them testable.

**Example:**
```python
# Retriever doesn't create its dependencies
retriever = Retriever(
    vector_store=vector_store,
    embedding_adapter=embedding_adapter,
    top_k=5
)
```

## 📦 Key Dependencies

### Backend (Python)

| Dependency | Purpose | Type |
|------------|---------|------|
| FastAPI | Web framework | Core |
| Uvicorn | ASGI server | Core |
| Pydantic | Data validation | Core |
| sentence-transformers | Local embeddings | Core |
| ChromaDB | Vector database | Core |
| PyPDF2 | PDF extraction | Documents |
| python-docx | Word extraction | Documents |
| Pillow | Image processing | OCR |
| pytesseract | OCR | OCR |
| pdf2image | PDF to images | OCR |
| SQLAlchemy | Database ORM | Database |
| pytest | Testing | Development |

### Frontend (Node.js)

| Dependency | Purpose | Type |
|------------|---------|------|
| React | UI library | Core |
| TypeScript | Type safety | Core |
| Vite | Build tool | Development |

## 🔧 Configuration Hierarchy

Configuration is loaded in this order (later overrides earlier):

1. **Defaults** in `src/config/config.py`
2. **YAML file** (`config.yaml`)
3. **Environment variables** (highest priority)

Example:
```python
# 1. Default
llm_provider = "ollama"

# 2. Overridden by config.yaml
llm:
  provider: "openai"

# 3. Overridden by environment variable
export LLM_PROVIDER=mock
```

## 🚀 Extension Points

Want to extend the system? Here are the main extension points:

### Add a New LLM Provider

1. Create `src/adapters/llm/your_provider.py`
2. Inherit from `LLMAdapter`
3. Implement `generate()` method
4. Register in `api/main.py`

### Add a New Document Type

1. Create `src/extractors/your_format.py`
2. Inherit from `BaseExtractor`
3. Implement `extract()` method
4. Register in `ExtractorFactory`

### Add a New Vector Store

1. Create `src/vectorstore/your_store.py`
2. Inherit from `VectorStore`
3. Implement required methods
4. Register in `api/main.py`

### Add a New API Endpoint

1. Add function in `api/main.py`
2. Create models in `api/models.py`
3. Add tests in `tests/test_e2e.py`

## 📚 Related Documentation

- **Getting Started:** [SETUP_GUIDE.md](../SETUP_GUIDE.md)
- **Contributing:** [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Architecture:** [docs/architecture.md](architecture.md)
- **API Reference:** http://localhost:8000/docs (when running)

## ❓ Questions?

- **General questions:** [GitHub Discussions](https://github.com/yourusername/rag_chatbot/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/yourusername/rag_chatbot/issues)
- **Contributing:** See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

**Happy coding!** 🚀

