# RAG Chatbot - Production-Quality Retrieval-Augmented Generation Pipeline

A modular, production-ready RAG (Retrieval-Augmented Generation) system with pluggable LLM providers, embedding models, and vector stores.

[![CI](https://github.com/yourusername/rag_chatbot/workflows/CI/badge.svg)](https://github.com/yourusername/rag_chatbot/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🔌 **Pluggable Architecture**: Swap LLM providers (Ollama, OpenAI), embeddings (local, OpenAI), and vector stores (Chroma, FAISS) via configuration
- 📚 **Multi-Format Support**: Index PDFs, DOCX, TXT, CSV, JSON files
- 🚀 **FastAPI Server**: RESTful API with auto-generated docs
- 🛠️ **CLI Tools**: Command-line scripts for indexing and querying
- 🧪 **Comprehensive Tests**: Unit and integration tests with >80% coverage
- 🐳 **Docker Ready**: Full docker-compose setup for development and production
- 📖 **Well Documented**: Architecture docs, API specs, and usage guides
- 🔒 **Production Best Practices**: Environment-based config, error handling, retry logic, health checks

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   FastAPI   │────▶│  Retriever   │────▶│ Vector Store│
│     API     │     │              │     │ (Chroma/    │
└─────────────┘     └──────────────┘     │  FAISS)     │
                            │             └─────────────┘
                            ▼                    ▲
                    ┌──────────────┐             │
                    │  Embedding   │─────────────┘
                    │   Adapter    │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │     LLM      │
                    │   Adapter    │
                    │ (Ollama/     │
                    │  OpenAI)     │
                    └──────────────┘
```

**Core Principles**:
- Single Responsibility: Each module has one clear purpose
- Dependency Injection: Components receive dependencies
- Interface Segregation: Abstract base classes define contracts
- Open/Closed: Extend functionality without modifying core

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & docker-compose (optional)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd rag_chatbot

# Install dependencies
pip install -r requirements.txt

# Configuration is ready! (Uses mock LLM by default for instant testing)
# config.yaml already exists with sensible defaults
# To use real AI later, edit config.yaml and switch to ollama or openai

# Create directories
mkdir -p data/sample data/chroma
```

### Index Sample Documents

```bash
# Create a sample document
echo "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It retrieves relevant documents and uses them as context for generating responses." > data/sample/rag_intro.txt

# Index it
python scripts/index_documents.py ./data/sample/
```

### Query the System

```bash
# Interactive mode
python scripts/query.py --interactive

# Single query
python scripts/query.py "What is RAG?" --show-sources
```

### Run API Server

```bash
# Start server
uvicorn api.main:app --reload

# Visit http://localhost:8000/docs for interactive API documentation
```

### Query via API

```bash
# Index a document
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is a subset of AI.", "metadata": {"source": "test"}}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 3}'
```

## 🧪 Test the API with Postman

- Import the collection at `docs/rag_chatbot_postman_collection.json` into Postman (File → Import).
- Set the `baseUrl` collection variable if your FastAPI server is not on `http://localhost:8000`.
- Use the provided requests to verify health, inspect stats, index text or files, and run chat queries.

## 🌐 React Operations Dashboard

A lightweight Vite + React frontend lives in `webapp/`. It lets you upload files, view the last indexed documents, and chat against the RAG pipeline.

### Prerequisites

- Node.js 18+ and npm (or Yarn / pnpm).

### Run in Development

```bash
cd webapp
npm install
cp env.example .env    # Optional: override the API base URL
npm run dev
```

By default, the app proxies requests to `http://localhost:8000`. Update `VITE_API_BASE_URL` in `.env` if your API runs elsewhere.

### Build for Production

```bash
npm run build
npm run preview   # Serves the production bundle locally
```

You can deploy the contents of `webapp/dist` behind any static site host (Vercel, Netlify, S3, etc.). Ensure the API endpoint is reachable and CORS is configured to allow the frontend origin.

## 🐳 Docker Quick Start

```bash
# Start all services (API, ChromaDB, Ollama)
docker-compose up -d

# Check health
curl http://localhost:8000/health

# Index documents
docker cp data/sample rag-api-1:/app/data/
docker exec rag-api-1 python scripts/index_documents.py /app/data/sample/

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

## ⚙️ Configuration

### LLM Providers

**Default: Mock LLM (No setup)**:
Already configured! Perfect for testing. Returns dummy responses instantly.

**Ollama (Local, Free)**:
```bash
# 1. Install Ollama from https://ollama.ai
# 2. Pull a model
ollama pull llama3.2:1b  # Fast, small (1.3GB)

# 3. Edit config.yaml
llm:
  provider: "ollama"  # Change from "mock"
```

**OpenAI (Cloud)**:
```bash
# 1. Get API key from https://platform.openai.com/api-keys
# 2. Set environment variable
export OPENAI_API_KEY=sk-your-key-here

# 3. Edit config.yaml
llm:
  provider: "openai"  # Change from "mock"
```

### Embedding Providers

**Local (sentence-transformers)**:
```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**OpenAI**:
```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

### Vector Stores

**ChromaDB** (Default):
```env
VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=./data/chroma
```

**FAISS** (High Performance):
```env
VECTORSTORE_PROVIDER=faiss
VECTORSTORE_PERSIST_PATH=./data/faiss
```

## 📁 Project Structure

```
rag_chatbot/
├── src/
│   ├── adapters/
│   │   ├── llm/              # LLM adapters (Ollama, OpenAI, Mock)
│   │   └── embedding/        # Embedding adapters (Local, OpenAI)
│   ├── extractors/           # Document extractors (PDF, DOCX, TXT, CSV)
│   ├── vectorstore/          # Vector stores (Chroma, FAISS, Memory)
│   ├── retriever/            # Retrieval logic
│   ├── config/               # Configuration management
│   └── utils/                # Chunking, prompts, helpers
├── api/
│   ├── main.py               # FastAPI application
│   └── models.py             # Pydantic models
├── scripts/
│   ├── index_documents.py    # CLI indexing tool
│   └── query.py              # CLI query tool
├── tests/                    # Comprehensive test suite
├── docs/                     # Documentation
├── docker-compose.yml        # Docker orchestration
├── Dockerfile                # Container definition
├── Makefile                  # Common tasks
└── requirements.txt          # Dependencies
```

## 🔧 Development

### Run Tests

```bash
make test

# Or directly
pytest tests/ -v --cov=src
```

### Lint and Format

```bash
make format  # Black + Ruff
make lint    # Ruff check
make typecheck  # MyPy
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## 📊 API Endpoints

- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /query` - Query with RAG
- `POST /index` - Index text or URL
- `POST /index/file` - Index uploaded file

Full API docs at `http://localhost:8000/docs`

## 🔄 Switching Providers

### Switch from Ollama to OpenAI

```bash
# Edit .env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here

# Restart server (no re-indexing needed)
```

### Switch Embedding Model

```bash
# Edit .env
EMBEDDING_PROVIDER=openai

# Re-index documents (different embeddings)
rm -rf ./data/chroma
python scripts/index_documents.py ./data/sample/
```

See [Switching Providers Guide](docs/switching_providers.md) for details.

## 🖼️ Future: Image & Video Support

The architecture includes hooks for media embeddings:

**Interface** (`src/adapters/embedding/base.py`):
```python
class MediaEmbeddingAdapter(ABC):
    @abstractmethod
    def embed_image(self, path: str) -> np.ndarray: pass
    
    @abstractmethod
    def embed_video(self, path: str) -> np.ndarray: pass
```

**To Add Support**:
1. Implement `MediaEmbeddingAdapter` using CLIP or ImageBind
2. Create `ImageExtractor` and `VideoExtractor`
3. Update `ExtractorFactory`

See [Embedding Guide](docs/embedding_guide.md) for implementation details.

## 📚 Documentation

- [Architecture Overview](docs/architecture.md) - System design and components
- [Getting Started](docs/getting_started.md) - Detailed setup guide
- [Switching Providers](docs/switching_providers.md) - Configuration guide
- [Embedding Guide](docs/embedding_guide.md) - Embedding models and optimization

## 🧪 Testing

```bash
# All tests
make test

# Specific test file
pytest tests/test_llm_adapters.py -v

# With coverage report
pytest --cov=src --cov-report=html
```

Coverage report: `htmlcov/index.html`

## 📦 Deployment

### Docker Production

```bash
# Build image
docker build -t rag-chatbot:latest .

# Run with environment
docker run -p 8000:8000 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-xxx \
  -v ./data:/app/data \
  rag-chatbot:latest
```

### Environment Variables

Required for OpenAI:
- `OPENAI_API_KEY`

Optional:
- `LLM_PROVIDER` (default: ollama)
- `EMBEDDING_PROVIDER` (default: local)
- `VECTORSTORE_PROVIDER` (default: chroma)

See `.env.example` for all options.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run checks: `make all`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Inspiration for modular design
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/rag_chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/rag_chatbot/discussions)

## 🗺️ Roadmap

- [x] Core RAG pipeline
- [x] Multiple LLM providers
- [x] Multiple embedding providers
- [x] Multiple vector stores
- [x] Document extractors
- [x] FastAPI server
- [x] CLI tools
- [x] Docker support
- [x] Comprehensive tests
- [ ] Streaming responses
- [ ] Hybrid search (BM25 + vector)
- [ ] Re-ranking
- [ ] Image/Video embeddings
- [ ] Conversation memory
- [ ] Web UI

---

**Built with ❤️ for production RAG applications**
