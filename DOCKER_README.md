# Docker Setup for RAG Chatbot

## Quick Start

### Start Everything

```bash
# Build and start all services (API + ChromaDB + Web UI)
docker-compose up --build -d

# View logs
docker-compose logs -f
```

### Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs  
- **Web UI**: http://localhost:5173
- **ChromaDB**: http://localhost:8001 (internal)

### Stop Services

```bash
docker-compose down
```

## Configuration

### Use Ollama (Local LLM)

1. **Install Ollama** on your host machine: https://ollama.ai

2. **Pull a model**:
   ```bash
   ollama pull llama3.2:1b
   ```

3. **Start containers**:
   ```bash
   docker-compose up -d
   ```

The API container will connect to Ollama on your host via `host.docker.internal:11434`

### Use OpenAI Instead

Create `.env` file:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

## Data Persistence

All data persists in the `./data` directory:

```
data/
├── chroma/          # Vector store (embeddings)
├── documents.db     # Document metadata
└── uploads/         # Temporary upload files
```

**This means**:
- ✅ Documents survive container restarts
- ✅ No need to re-index after `docker-compose down`
- ✅ Can backup by copying `data/` folder

## Development

### Hot Reload

Source code is mounted, so changes to Python files auto-reload:

```bash
# Edit src/adapters/llm/ollama.py
# API automatically restarts with changes

# Edit webapp/src/App.tsx  
# Vite hot-reloads the UI
```

### View Logs

```bash
# All services
docker-compose logs -f

# Just API
docker-compose logs -f rag-api

# Just webapp
docker-compose logs -f webapp
```

### Execute Commands in Container

```bash
# Index documents
docker-compose exec rag-api python scripts/index_documents.py /app/data/sample/

# Open Python shell
docker-compose exec rag-api python

# Check health
docker-compose exec rag-api curl http://localhost:8000/health
```

## Troubleshooting

### API Can't Connect to Ollama

**Error**: `Cannot connect to Ollama at http://host.docker.internal:11434`

**Fix**:
1. Verify Ollama is running: `ollama list`
2. Check model is pulled: `ollama pull llama3.2:1b`
3. Test connection from container:
   ```bash
   docker-compose exec rag-api curl http://host.docker.internal:11434/api/tags
   ```

### Model Not Found

**Error**: `Model 'llama3.2:1b' not found in Ollama`

**Fix**:
```bash
# On host machine
ollama pull llama3.2:1b

# Restart API
docker-compose restart rag-api
```

### Port Already in Use

**Error**: `port is already allocated`

**Fix**:
```bash
# Find what's using port 8000
# Windows
netstat -ano | findstr :8000

# Mac/Linux  
lsof -ti:8000

# Kill it or change port in docker-compose.yml
ports:
  - "8080:8000"  # Use 8080 instead
```

### Database Locked

**Error**: `database is locked`

**Fix**:
```bash
docker-compose down
rm data/chroma/chroma.sqlite3-wal
rm data/chroma/chroma.sqlite3-shm
docker-compose up -d
```

### Rebuild from Scratch

```bash
# Stop everything
docker-compose down -v

# Remove images
docker-compose down --rmi all -v

# Clear data (CAUTION: deletes all documents!)
rm -rf data/chroma/* data/documents.db

# Rebuild
docker-compose up --build -d
```

## Common Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart a service
docker-compose restart rag-api

# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# Rebuild images
docker-compose build

# Start without webapp
docker-compose up -d rag-api chroma

# Run npm install in webapp
docker-compose exec webapp npm install

# Access container shell
docker-compose exec rag-api bash
```

## Environment Variables

Available in `.env` file:

```env
# LLM
LLM_PROVIDER=ollama              # ollama or openai
OLLAMA_MODEL=llama3.2:1b         # Model to use
OPENAI_API_KEY=sk-xxx            # If using OpenAI

# Embedding  
EMBEDDING_PROVIDER=local         # local or openai

# Vector Store
VECTORSTORE_PROVIDER=chroma      # chroma, faiss, or memory
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│   webapp    │────▶│   rag-api   │
│   :5173     │     │   (Vite)    │     │   :8000     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                    ┌──────────────────────────┴─────┐
                    │                                 │
                    ▼                                 ▼
            ┌──────────────┐              ┌──────────────┐
            │    chroma    │              │   Ollama     │
            │    :8001     │              │  (host)      │
            └──────────────┘              │  :11434      │
                                          └──────────────┘
```

## Volumes

```yaml
volumes:
  - ./data:/app/data              # Persistent storage
  - ./src:/app/src                # Hot reload (API)
  - ./api:/app/api                # Hot reload (API)
  - ./config.yaml:/app/config.yaml # Config file
  - ./webapp/src:/app/src         # Hot reload (webapp)
```

## Health Checks

Built-in health monitoring:

```bash
# Check container health
docker-compose ps

# Manual health check
curl http://localhost:8000/health
```

Health check runs every 30 seconds automatically.

---

**That's it!** Run `docker-compose up -d` and you're good to go! 🚀

