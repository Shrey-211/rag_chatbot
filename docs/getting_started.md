# Getting Started

## Prerequisites

- Python 3.11 or higher
- Docker and docker-compose (optional, for containerized deployment)
- Git

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd rag_chatbot
```

### 2. Install Dependencies

#### Option A: Using Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Option B: Using Make

```bash
make install
```

### 3. Configuration

Copy example configuration files:

```bash
cp .env.example .env
cp config.example.yaml config.yaml
```

Edit `.env` and set your preferences:

```env
# Choose your LLM provider
LLM_PROVIDER=ollama  # or openai, mock

# Ollama settings (if using ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# OpenAI settings (if using openai)
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Embedding provider
EMBEDDING_PROVIDER=local  # or openai
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector store
VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=./data/chroma
```

## Quick Start (Local)

### 1. Prepare Sample Data

Create a sample document:

```bash
mkdir -p data/sample
echo "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It first retrieves relevant documents and then uses them as context for generating responses." > data/sample/rag_intro.txt
```

### 2. Index Documents

```bash
python scripts/index_documents.py ./data/sample/
```

Output:
```
INFO - Starting indexing with configuration:
INFO -   Embedding provider: local
INFO -   Vector store provider: chroma
INFO - Loading embedding model: all-MiniLM-L6-v2 on cpu
INFO - Model loaded. Embedding dimension: 384
...
INFO - Indexing completed!
INFO -   Files processed: 1
INFO -   Total chunks indexed: 3
```

### 3. Query the System

#### Interactive Mode

```bash
python scripts/query.py --interactive
```

```
RAG Interactive Query Mode
Type your questions below. Type 'exit' or 'quit' to exit.

Query: What is RAG?
Generating answer...
Answer: RAG stands for Retrieval-Augmented Generation...
```

#### Single Query

```bash
python scripts/query.py "What is RAG?" --show-sources
```

### 4. Run API Server

```bash
python api/main.py
# Or using uvicorn directly:
uvicorn api.main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

## Quick Start (Docker)

### 1. Start Services

```bash
docker-compose up -d
```

This starts:
- RAG API server (port 8000)
- ChromaDB (port 8001)
- Ollama (port 11434, optional)

### 2. Check Health

```bash
curl http://localhost:8000/health
```

### 3. Index Documents

```bash
# Copy sample data into container
docker cp data/sample rag-api-1:/app/data/

# Run indexing
docker exec rag-api-1 python scripts/index_documents.py /app/data/sample/
```

### 4. Query via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 3, "include_sources": true}'
```

## Using the API

### Index Text

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text here",
    "metadata": {"source": "manual"}
  }'
```

### Index File

```bash
curl -X POST http://localhost:8000/index/file \
  -F "file=@/path/to/document.pdf"
```

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Your question here",
    "top_k": 5,
    "include_sources": true
  }'
```

### Get Statistics

```bash
curl http://localhost:8000/stats
```

## Development Workflow

### 1. Setup Development Environment

```bash
make setup
```

This creates `.env`, `config.yaml`, and necessary directories.

### 2. Run Tests

```bash
make test
```

### 3. Lint and Format

```bash
make format
make lint
```

### 4. Type Check

```bash
make typecheck
```

### 5. Run All Checks

```bash
make all
```

## Common Tasks

### Switching to OpenAI

1. Edit `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_PROVIDER=openai
```

2. Restart services

### Using FAISS Instead of Chroma

1. Edit `.env`:
```env
VECTORSTORE_PROVIDER=faiss
VECTORSTORE_PERSIST_PATH=./data/faiss
```

2. Re-index documents (FAISS uses different storage)

### Indexing Large Document Collections

```bash
# Recursive indexing
python scripts/index_documents.py /path/to/documents --recursive

# With custom batch size
python scripts/index_documents.py /path/to/documents --batch-size 50
```

### Clearing Vector Store

```bash
# For Chroma
rm -rf ./data/chroma

# For FAISS
rm -rf ./data/faiss
```

## Troubleshooting

### Ollama Connection Error

**Problem**: `Ollama health check failed`

**Solution**:
1. Ensure Ollama is running: `ollama serve`
2. Pull required model: `ollama pull llama2`
3. Check connection: `curl http://localhost:11434/api/tags`

### Out of Memory

**Problem**: Large model crashes

**Solution**:
1. Use smaller embedding model: `EMBEDDING_MODEL=all-MiniLM-L6-v2`
2. Reduce batch size: `--batch-size 10`
3. Use FAISS instead of Chroma for large datasets

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure you're in the project root
pwd  # Should show .../rag_chatbot

# Reinstall dependencies
pip install -r requirements.txt
```

### Empty Search Results

**Problem**: Queries return no results

**Solution**:
1. Check documents are indexed: `curl http://localhost:8000/stats`
2. Verify vector store path exists
3. Re-index documents
4. Lower `min_relevance_score` in config

## Next Steps

- [Architecture Overview](architecture.md)
- [Switching Providers](switching_providers.md)
- [Embedding Guide](embedding_guide.md)

