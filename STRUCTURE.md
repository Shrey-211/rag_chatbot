# Project Structure

Complete file tree for the RAG Chatbot repository.

```
rag_chatbot/
│
├── README.md                           # Main documentation and quickstart
├── STRUCTURE.md                        # This file - complete file tree
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project configuration and tool settings
├── Makefile                           # Common development tasks
├── Dockerfile                         # Container definition
├── docker-compose.yml                 # Multi-container orchestration
│
├── .env.example                       # Example environment variables (blocked by .gitignore)
├── config.example.yaml                # Example YAML configuration
├── .gitignore                         # Git ignore patterns
├── .pre-commit-config.yaml           # Pre-commit hooks configuration
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI/CD pipeline
│
├── src/                               # Main source code
│   ├── __init__.py
│   │
│   ├── adapters/                      # Adapter pattern implementations
│   │   ├── __init__.py
│   │   │
│   │   ├── llm/                       # LLM provider adapters
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Abstract LLMAdapter interface
│   │   │   ├── ollama.py             # Ollama local LLM adapter
│   │   │   ├── openai.py             # OpenAI API adapter
│   │   │   └── mock.py               # Mock adapter for testing
│   │   │
│   │   └── embedding/                 # Embedding provider adapters
│   │       ├── __init__.py
│   │       ├── base.py               # Abstract EmbeddingAdapter interface
│   │       ├── local.py              # Local sentence-transformers adapter
│   │       └── openai.py             # OpenAI embeddings adapter
│   │
│   ├── extractors/                    # Document extraction
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract DocumentExtractor + Factory
│   │   ├── txt.py                    # Plain text extractor
│   │   ├── pdf.py                    # PDF extractor (PyPDF2)
│   │   ├── docx.py                   # Word document extractor
│   │   └── table.py                  # CSV/JSON/JSONL extractor
│   │
│   ├── vectorstore/                   # Vector database implementations
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract VectorStore interface
│   │   ├── chroma.py                 # ChromaDB implementation
│   │   ├── faiss.py                  # FAISS implementation
│   │   └── memory.py                 # In-memory implementation (testing)
│   │
│   ├── retriever/                     # Retrieval logic
│   │   ├── __init__.py
│   │   └── retriever.py              # Retriever class (query → results)
│   │
│   ├── config/                        # Configuration management
│   │   ├── __init__.py
│   │   └── config.py                 # Config loader (YAML + env vars)
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── chunking.py               # Text chunking with overlap
│       └── prompts.py                # Prompt templates
│
├── api/                               # FastAPI application
│   ├── __init__.py
│   ├── main.py                       # FastAPI app with endpoints
│   └── models.py                     # Pydantic request/response models
│
├── scripts/                           # CLI scripts
│   ├── index_documents.py            # Document indexing CLI
│   └── query.py                      # Query CLI (interactive + single)
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_llm_adapters.py          # LLM adapter tests
│   ├── test_embedding_adapters.py    # Embedding adapter tests
│   ├── test_extractors.py            # Document extractor tests
│   ├── test_vectorstore.py           # Vector store tests
│   └── test_e2e.py                   # End-to-end integration tests
│
├── docs/                              # Documentation
│   ├── architecture.md               # System architecture and design
│   ├── getting_started.md            # Setup and quickstart guide
│   ├── switching_providers.md        # Provider configuration guide
│   └── embedding_guide.md            # Embedding models and optimization
│
├── examples/                          # Example scripts
│   └── quickstart.py                 # Complete RAG pipeline example
│
└── data/                              # Data directory (created at runtime)
    ├── sample/                        # Sample documents for testing
    │   └── rag_introduction.txt      # Example document about RAG
    ├── chroma/                        # ChromaDB persistence (git-ignored)
    ├── faiss/                         # FAISS index storage (git-ignored)
    └── uploads/                       # Temporary file uploads (git-ignored)
```

## File Counts

- **Total Files**: ~55 files
- **Source Files**: 30+ Python modules
- **Test Files**: 5 test modules
- **Documentation**: 4 markdown guides
- **Configuration**: 7 config files
- **Scripts**: 2 CLI tools

## Key Files by Purpose

### Entry Points

- `api/main.py` - FastAPI server entry point
- `scripts/index_documents.py` - CLI indexing
- `scripts/query.py` - CLI querying
- `examples/quickstart.py` - Quickstart example

### Configuration

- `.env.example` - Environment variable template
- `config.example.yaml` - YAML configuration template
- `src/config/config.py` - Configuration loader

### Core Interfaces

- `src/adapters/llm/base.py` - LLM adapter interface
- `src/adapters/embedding/base.py` - Embedding adapter interface
- `src/vectorstore/base.py` - Vector store interface
- `src/extractors/base.py` - Document extractor interface

### Implementations

**LLM Adapters**:
- `ollama.py` - Local Ollama integration
- `openai.py` - OpenAI Chat Completions
- `mock.py` - Testing mock

**Embedding Adapters**:
- `local.py` - sentence-transformers (local)
- `openai.py` - OpenAI embeddings API

**Vector Stores**:
- `chroma.py` - ChromaDB (default, persistent)
- `faiss.py` - FAISS (high performance)
- `memory.py` - In-memory (testing)

**Extractors**:
- `txt.py` - Text files (.txt, .md, .log)
- `pdf.py` - PDF files
- `docx.py` - Word documents
- `table.py` - CSV, JSON, JSONL

### Testing

- `test_llm_adapters.py` - Tests for mock LLM
- `test_embedding_adapters.py` - Tests for local embeddings
- `test_extractors.py` - Tests for extractors + factory
- `test_vectorstore.py` - Tests for in-memory store
- `test_e2e.py` - Full pipeline integration tests

### DevOps

- `Dockerfile` - Container image
- `docker-compose.yml` - Multi-service setup
- `.github/workflows/ci.yml` - CI/CD pipeline
- `Makefile` - Development commands
- `.pre-commit-config.yaml` - Git hooks

## Adding New Components

### New LLM Provider

1. Create `src/adapters/llm/your_provider.py`
2. Inherit from `LLMAdapter` (base.py)
3. Implement: `generate()`, `stream_generate()`, `token_usage()`, `health_check()`, `get_model_info()`
4. Register in `api/main.py` `create_llm_adapter()`
5. Add config to `config.example.yaml`

### New Embedding Provider

1. Create `src/adapters/embedding/your_provider.py`
2. Inherit from `EmbeddingAdapter` (base.py)
3. Implement: `embed_texts()`, `embed_file()`, `get_embedding_dimension()`, `health_check()`
4. Register in `api/main.py` `create_embedding_adapter()`

### New Vector Store

1. Create `src/vectorstore/your_store.py`
2. Inherit from `VectorStore` (base.py)
3. Implement: `upsert()`, `query()`, `delete()`, `persist()`, `count()`, `get_by_ids()`
4. Register in `api/main.py` `create_vector_store()`

### New Document Type

1. Create `src/extractors/your_type.py`
2. Inherit from `DocumentExtractor` (base.py)
3. Implement: `extract()`, `supports()`
4. Add to `ExtractorFactory` default list in `base.py`

### New Test Suite

1. Create `tests/test_your_feature.py`
2. Use pytest fixtures for setup
3. Test happy path + edge cases
4. Run: `pytest tests/test_your_feature.py -v`

## Where to Find Things

**Switching LLM**: Edit `.env` → `LLM_PROVIDER=ollama|openai|mock`

**Switching Embeddings**: Edit `.env` → `EMBEDDING_PROVIDER=local|openai` (requires re-indexing)

**Switching Vector Store**: Edit `.env` → `VECTORSTORE_PROVIDER=chroma|faiss|memory`

**Prompt Templates**: `src/utils/prompts.py` or `config.yaml` → `prompts.rag_template`

**Chunking Strategy**: `src/utils/chunking.py` → `chunk_text()` function

**API Endpoints**: `api/main.py` → `/query`, `/index`, `/index/file`, `/health`, `/stats`

**Image/Video Support (Future)**: `src/adapters/embedding/base.py` → `MediaEmbeddingAdapter` interface

## Dependencies Summary

**Core**:
- fastapi, uvicorn - API server
- pydantic, pydantic-settings - Configuration
- sentence-transformers, torch - Local embeddings
- chromadb, faiss-cpu - Vector stores
- openai, httpx - API clients

**Document Processing**:
- PyPDF2 - PDF extraction
- python-docx - Word extraction
- pandas - CSV/JSON extraction

**Development**:
- pytest, pytest-cov - Testing
- black, ruff - Linting/formatting
- mypy - Type checking
- pre-commit - Git hooks

**Utilities**:
- tenacity - Retry logic
- tqdm - Progress bars
- python-dotenv, pyyaml - Configuration

See `requirements.txt` for complete list with versions.

