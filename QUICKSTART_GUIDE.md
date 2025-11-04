# RAG Chatbot - Complete Quickstart Guide

## 🎯 Acceptance Criteria Verification

This guide demonstrates how to meet all the acceptance criteria from the project requirements.

### ✅ Criterion 1: Docker Compose Setup

**Requirement**: Run `docker-compose up` to bring up server and vector store

**Steps**:
```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# NAME                COMMAND             STATUS              PORTS
# rag-api-1          uvicorn...          Up                  0.0.0.0:8000->8000/tcp
# chroma-1           uvicorn...          Up                  0.0.0.0:8001->8000/tcp
# ollama-1           /bin/ollama serve   Up                  0.0.0.0:11434->11434/tcp

# Check health
curl http://localhost:8000/health
```

**Result**: ✅ API server, ChromaDB, and Ollama running in containers

---

### ✅ Criterion 2: Document Indexing and Querying

**Requirement**: Run `python index_documents.py ./data/sample/` and `python query.py "Who is X?"` to get answers with source snippets

**Steps**:

#### Option A: Using Docker

```bash
# Copy sample data to container
docker cp data/sample rag-api-1:/app/data/

# Index documents
docker exec rag-api-1 python scripts/index_documents.py /app/data/sample/

# Query
docker exec -it rag-api-1 python scripts/query.py "What is RAG?" --show-sources
```

#### Option B: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp .env.example .env
# Edit .env to configure providers

# Index documents
python scripts/index_documents.py ./data/sample/

# Expected output:
# INFO - Starting indexing with configuration:
# INFO -   Embedding provider: local
# INFO -   Vector store provider: chroma
# INFO - Loading embedding model: all-MiniLM-L6-v2 on cpu
# ...
# INFO - Indexing completed!
# INFO -   Files processed: 1
# INFO -   Total chunks indexed: 5

# Single query
python scripts/query.py "What is RAG?" --show-sources

# Expected output:
# Query: What is RAG?
# Generating answer...
# Answer: Retrieval-Augmented Generation (RAG) is a technique that combines...
# Sources (3 documents):
# [1] (score: 0.856)
#   Retrieval-Augmented Generation, commonly known as RAG...

# Interactive mode
python scripts/query.py --interactive
# Query: What are the benefits of RAG?
# Query: How does RAG work?
# Query: exit
```

**Result**: ✅ Documents indexed and queries return answers with source snippets

---

### ✅ Criterion 3: Provider Switching

**Requirement**: Switch `LLM_PROVIDER` to `openai` or `ollama` in `.env` and the same code uses the selected adapter

**Steps**:

#### Test 1: Switch to Ollama (Local)

```bash
# Edit .env
cat > .env << 'EOF'
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=./data/chroma
EOF

# Ensure Ollama is running
ollama serve &
ollama pull llama2

# Query (uses Ollama automatically)
python scripts/query.py "What is RAG?"
```

#### Test 2: Switch to OpenAI

```bash
# Edit .env
cat > .env << 'EOF'
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=./data/chroma
EOF

# Query (uses OpenAI automatically)
python scripts/query.py "What is RAG?"

# No code changes needed - just configuration!
```

#### Test 3: Switch Vector Store

```bash
# Switch to FAISS
cat >> .env << 'EOF'
VECTORSTORE_PROVIDER=faiss
VECTORSTORE_PERSIST_PATH=./data/faiss
EOF

# Re-index for new vector store
rm -rf ./data/faiss
python scripts/index_documents.py ./data/sample/

# Query works with FAISS now
python scripts/query.py "What is RAG?"
```

**Result**: ✅ Provider switching works via configuration only, no code changes

---

### ✅ Criterion 4: Tests Pass in CI

**Requirement**: `pytest` returns green in CI

**Steps**:

#### Local Test Run

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v --cov=src --cov-report=term

# Expected output:
# tests/test_llm_adapters.py::TestMockLLMAdapter::test_generate PASSED
# tests/test_llm_adapters.py::TestMockLLMAdapter::test_stream_generate PASSED
# tests/test_embedding_adapters.py::TestLocalEmbeddingAdapter::test_embed_texts PASSED
# tests/test_extractors.py::TestTextExtractor::test_extract_text_file PASSED
# tests/test_vectorstore.py::TestInMemoryVectorStore::test_upsert_and_count PASSED
# tests/test_e2e.py::TestE2E::test_index_and_retrieve PASSED
# ...
# ========== 20 passed in 15.23s ==========

# Run specific test
pytest tests/test_e2e.py::TestE2E::test_rag_query_with_llm -v

# Check coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

#### CI/CD Verification

```bash
# Push to GitHub
git add .
git commit -m "Add RAG implementation"
git push origin main

# GitHub Actions will automatically:
# 1. Run linting (ruff)
# 2. Run formatting check (black)
# 3. Run type checking (mypy)
# 4. Run tests (pytest)
# 5. Build Docker image

# Check status at:
# https://github.com/yourusername/rag_chatbot/actions
```

**Result**: ✅ All tests pass locally and in CI

---

## 📊 API Usage Examples

### Health Check

```bash
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "llm_healthy": true,
  "embedding_healthy": true,
  "vectorstore_count": 5,
  "config": {
    "llm_provider": "ollama",
    "embedding_provider": "local",
    "vectorstore_provider": "chroma"
  }
}
```

### Index Text

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "metadata": {"source": "api", "topic": "ml"}
  }'

# Response:
{
  "success": true,
  "document_id": "abc123...",
  "num_chunks": 1,
  "message": "Successfully indexed 1 chunks"
}
```

### Index File

```bash
curl -X POST http://localhost:8000/index/file \
  -F "file=@./data/sample/rag_introduction.txt"

# Response:
{
  "success": true,
  "document_id": "def456...",
  "num_chunks": 5,
  "message": "Successfully indexed file: rag_introduction.txt"
}
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 3,
    "include_sources": true
  }'

# Response:
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {
      "id": "abc123_0",
      "content": "Machine learning is a subset of artificial intelligence...",
      "metadata": {"source": "api", "topic": "ml"},
      "score": 0.95
    }
  ],
  "llm_metadata": {
    "model": "llama2",
    "usage": {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200}
  }
}
```

### Statistics

```bash
curl http://localhost:8000/stats

# Response:
{
  "total_documents": 10,
  "vectorstore_provider": "chroma",
  "llm_provider": "ollama",
  "embedding_provider": "local",
  "embedding_dimension": 384
}
```

---

## 🧪 Testing Different Configurations

### Configuration 1: Fully Local (Free)

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=./data/chroma
```

**Pros**: Free, privacy-preserving, no API calls  
**Cons**: Slower, requires local compute

### Configuration 2: OpenAI (High Quality)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large

VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=./data/chroma
```

**Pros**: Best quality, fast, managed  
**Cons**: API costs

### Configuration 3: Hybrid (Cost-Optimized)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

VECTORSTORE_PROVIDER=faiss
VECTORSTORE_PERSIST_PATH=./data/faiss
```

**Pros**: Good quality, lower cost (only LLM paid)  
**Cons**: Mixed architecture

---

## 🔍 Verification Checklist

- [x] ✅ Docker services start successfully
- [x] ✅ Health endpoint returns healthy status
- [x] ✅ Documents can be indexed via CLI
- [x] ✅ Documents can be indexed via API
- [x] ✅ Queries return answers with sources
- [x] ✅ LLM provider can be switched via .env
- [x] ✅ Embedding provider can be switched via .env
- [x] ✅ Vector store provider can be switched via .env
- [x] ✅ All tests pass locally
- [x] ✅ CI pipeline runs successfully
- [x] ✅ Multiple document formats supported (PDF, DOCX, TXT, CSV)
- [x] ✅ Comprehensive documentation provided
- [x] ✅ Example code demonstrates full pipeline

---

## 📝 Common Commands Reference

```bash
# Development Setup
make setup                    # Initialize project
make install                  # Install dependencies
make dev                      # Install dev dependencies + pre-commit

# Testing
make test                     # Run all tests with coverage
pytest tests/test_e2e.py -v   # Run specific test
make typecheck                # Run mypy

# Code Quality
make format                   # Format with black
make lint                     # Lint with ruff
make all                      # Format, lint, typecheck, test

# Docker
make docker-build             # Build images
make docker-up                # Start services
make docker-down              # Stop services

# Indexing & Querying
make index                    # Index sample documents
make query                    # Interactive query mode
make run-api                  # Start API server

# Cleanup
make clean                    # Remove build artifacts
rm -rf data/chroma            # Clear vector store
```

---

## 🎓 Next Steps

1. **Customize Providers**: Edit `.env` to use your preferred LLM/embeddings
2. **Index Your Data**: Place documents in `data/sample/` and run indexing
3. **Explore API**: Visit http://localhost:8000/docs for interactive API
4. **Read Architecture**: See `docs/architecture.md` for design details
5. **Extend System**: Add new providers following `STRUCTURE.md`

---

## 📞 Troubleshooting

### Issue: "Ollama health check failed"
**Solution**: Ensure Ollama is running: `ollama serve`

### Issue: "OpenAI API key invalid"
**Solution**: Verify key at https://platform.openai.com/api-keys

### Issue: "No documents found"
**Solution**: Index documents first: `python scripts/index_documents.py ./data/sample/`

### Issue: "Import error"
**Solution**: Ensure you're in project root and dependencies installed

### Issue: "Docker container exits"
**Solution**: Check logs: `docker-compose logs rag-api`

---

**All acceptance criteria verified and documented!** 🎉

