# RAG Chatbot Architecture

## Overview

This RAG (Retrieval-Augmented Generation) chatbot is built with a modular, adapter-based architecture that allows easy swapping of LLM providers, embedding models, and vector stores.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  /query  /index  /index/file  /health  /stats               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   RAG Orchestration                          │
│  - Query Processing                                          │
│  - Document Indexing                                         │
│  - Prompt Management                                         │
└──────┬──────────────┬──────────────────┬────────────────────┘
       │              │                  │
┌──────▼──────┐ ┌─────▼──────┐ ┌────────▼────────┐
│  Retriever  │ │   LLM      │ │   Document      │
│             │ │  Adapter   │ │   Extractors    │
│  - Search   │ │            │ │                 │
│  - Ranking  │ │ - Ollama   │ │ - PDF           │
│  - Format   │ │ - OpenAI   │ │ - DOCX          │
│             │ │ - Mock     │ │ - TXT           │
└──────┬──────┘ └────────────┘ │ - CSV/JSON      │
       │                        └─────────────────┘
┌──────▼──────────────────────────────┐
│         Vector Store                │
│  - Chroma (default)                 │
│  - FAISS                            │
│  - In-Memory (testing)              │
└──────┬──────────────────────────────┘
       │
┌──────▼──────────────────────────────┐
│      Embedding Adapter              │
│  - Local (sentence-transformers)    │
│  - OpenAI                           │
└─────────────────────────────────────┘
```

## Core Components

### 1. Adapters

#### LLM Adapters (`src/adapters/llm/`)
- **Base Interface**: `LLMAdapter` abstract class
- **Methods**:
  - `generate(prompt, **kwargs)` - Synchronous generation
  - `stream_generate(prompt, callback, **kwargs)` - Streaming generation
  - `token_usage(response)` - Extract token usage
  - `health_check()` - Service health verification
  - `get_model_info()` - Model metadata

- **Implementations**:
  - `OllamaAdapter` - Local Ollama API
  - `OpenAIAdapter` - OpenAI Chat Completions
  - `MockLLMAdapter` - Testing mock

#### Embedding Adapters (`src/adapters/embedding/`)
- **Base Interface**: `EmbeddingAdapter` abstract class
- **Methods**:
  - `embed_texts(texts)` - Batch text embedding
  - `embed_file(path)` - File embedding
  - `embed_batch(iterator)` - Streaming batch processing
  - `get_embedding_dimension()` - Dimension info
  - `health_check()` - Service health

- **Implementations**:
  - `LocalTextEmbeddingAdapter` - sentence-transformers (default: all-MiniLM-L6-v2)
  - `OpenAIEmbeddingAdapter` - OpenAI embeddings API

### 2. Vector Stores (`src/vectorstore/`)

- **Base Interface**: `VectorStore` abstract class
- **Methods**:
  - `upsert(ids, embeddings, documents, metadatas)` - Add/update documents
  - `query(embedding, top_k, filter_dict)` - Similarity search
  - `delete(ids, filter_dict)` - Remove documents
  - `persist()` - Save to disk
  - `count()` - Total documents
  - `get_by_ids(ids)` - Retrieve by ID

- **Implementations**:
  - `ChromaVectorStore` - ChromaDB (persistent, recommended)
  - `FAISSVectorStore` - Facebook FAISS (fast, memory-efficient)
  - `InMemoryVectorStore` - Simple in-memory (testing only)

### 3. Document Extractors (`src/extractors/`)

- **Base Interface**: `DocumentExtractor` abstract class
- **Methods**:
  - `extract(file_path)` - Extract text from file
  - `supports(file_path)` - Check file type support

- **Implementations**:
  - `TextExtractor` - .txt, .md, .log files
  - `PDFExtractor` - PDF files (PyPDF2)
  - `DocxExtractor` - Microsoft Word files
  - `TableExtractor` - CSV, JSON, JSONL

- **Factory Pattern**: `ExtractorFactory` auto-selects extractor by file extension

### 4. Retriever (`src/retriever/`)

The `Retriever` orchestrates the retrieval pipeline:
1. Embed query using `EmbeddingAdapter`
2. Search `VectorStore` for similar documents
3. Filter by minimum relevance score
4. Format context for LLM

### 5. Configuration (`src/config/`)

Configuration is loaded from:
1. YAML file (`config.yaml`)
2. Environment variables (override YAML)
3. Defaults (fallback)

Key configurations:
- `llm.provider`: ollama | openai | mock
- `embedding.provider`: local | openai
- `vectorstore.provider`: chroma | faiss | memory
- `retrieval.chunk_size`: Text chunk size
- `retrieval.top_k`: Number of results

### 6. Utilities (`src/utils/`)

- **Chunking** (`chunking.py`): Split text into overlapping chunks
- **Prompts** (`prompts.py`): Template system for prompts with variable substitution

## Data Flow

### Indexing Pipeline

```
Document → Extractor → Text → Chunker → Chunks
                                           ↓
                                   EmbeddingAdapter
                                           ↓
                                      Embeddings
                                           ↓
                                      VectorStore
```

### Query Pipeline

```
Query → EmbeddingAdapter → Query Embedding
                                ↓
                         VectorStore.query()
                                ↓
                          Search Results
                                ↓
                         Format Context
                                ↓
                    Build Prompt (Template)
                                ↓
                          LLM Adapter
                                ↓
                           Response
```

## Extensibility Points

### Adding New LLM Provider

1. Create new class in `src/adapters/llm/`
2. Inherit from `LLMAdapter`
3. Implement all abstract methods
4. Register in `api/main.py` `create_llm_adapter()`
5. Add config section to `config.yaml`

### Adding New Embedding Provider

Similar to LLM, implement `EmbeddingAdapter` interface.

### Adding Media Embedding (Future)

1. Implement `MediaEmbeddingAdapter` interface (defined in `src/adapters/embedding/base.py`)
2. Add methods: `embed_image()`, `embed_video()`, `embed_frames()`
3. Use CLIP, ImageBind, or similar multi-modal models
4. Update extractors to handle image/video files

### Adding New Document Type

1. Create extractor in `src/extractors/`
2. Inherit from `DocumentExtractor`
3. Implement `extract()` and `supports()` methods
4. Add to `ExtractorFactory` default list

## Technology Stack

- **Framework**: FastAPI (async API server)
- **LLM**: Ollama (local), OpenAI (remote)
- **Embeddings**: sentence-transformers (local), OpenAI (remote)
- **Vector DB**: ChromaDB, FAISS
- **Document Processing**: PyPDF2, python-docx, pandas
- **Testing**: pytest, pytest-asyncio
- **CI/CD**: GitHub Actions
- **Containerization**: Docker, docker-compose

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Injection**: Components receive dependencies, not create them
3. **Interface Segregation**: Abstract base classes define contracts
4. **Open/Closed**: Open for extension, closed for modification
5. **Test Coverage**: Unit and integration tests for all components
6. **Configuration over Code**: Behavior controlled via config files
7. **Fail Fast**: Early validation and clear error messages

## Performance Considerations

- **Batch Processing**: Embeddings processed in batches
- **Vector Store Selection**: 
  - Chroma: Good for persistent, moderate-scale
  - FAISS: Faster for large-scale, in-memory
  - In-Memory: Only for testing
- **Chunking Strategy**: Overlap ensures context preservation
- **Caching**: Vector stores persist embeddings (no re-computation)

## Security

- API keys loaded from environment variables
- `.env` file in `.gitignore`
- No secrets in code or config files
- CORS configured in `config.yaml`
- File upload size limits (FastAPI default)

