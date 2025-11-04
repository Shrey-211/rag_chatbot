# Ollama Setup Guide for Your RAG Chatbot

## ✅ What's Already Configured

Your RAG system is **already optimized** to connect to Ollama running on your local laptop:
- Docker container → `http://host.docker.internal:11434` → Your laptop's Ollama

## 📦 Models You Need to Install

### Model Summary

| Component | Type | Installation | Download Location |
|-----------|------|--------------|-------------------|
| **LLM (Ollama)** | Required | Manual (you install) | Your laptop |
| **Embeddings** | Automatic | Auto-downloads | Inside Docker container |

---

## 🚀 Step-by-Step Setup

### Step 1: Install Ollama

```bash
# Download from: https://ollama.ai
# or on Windows with winget:
winget install Ollama.Ollama
```

### Step 2: Pull Your LLM Model

**For RTX 4050 6GB VRAM, I recommend `mistral`:**

```bash
# Recommended for your GPU
ollama pull mistral

# Or other options:
ollama pull llama2        # Default (7B, ~4GB VRAM)
ollama pull phi           # Smallest (2.7B, ~2GB VRAM)
ollama pull llama2:13b    # Larger (might be tight on 6GB)
```

### Step 3: Start Ollama Service

```bash
ollama serve
```

Keep this terminal open - Ollama needs to be running!

### Step 4: Verify Ollama is Running

```bash
# List installed models
ollama list

# Test the model
ollama run mistral "Hello, how are you?"
```

---

## 🔧 Switching Models

If you want to use a different model:

### Option 1: Change in Docker (Easiest)

Edit `docker-compose.yml` and change the OLLAMA_MODEL:

```yaml
environment:
  - OLLAMA_MODEL=mistral  # Change this line
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

### Option 2: Set Environment Variable

```bash
# Windows PowerShell
$env:OLLAMA_MODEL="mistral"
docker-compose up -d
```

---

## 📊 Model Comparison

### For Your RTX 4050 6GB VRAM:

| Model | Size | VRAM | Speed | Quality | Recommendation |
|-------|------|------|-------|---------|----------------|
| `phi` | 2.7B | ~2GB | ⚡⚡⚡ | ⭐⭐⭐ | Fast, lower quality |
| `mistral` | 7B | ~4GB | ⚡⚡ | ⭐⭐⭐⭐ | **✅ Best for 6GB** |
| `llama2` | 7B | ~4GB | ⚡⚡ | ⭐⭐⭐⭐ | Good default |
| `llama2:13b` | 13B | ~8GB | ⚡ | ⭐⭐⭐⭐⭐ | ⚠️ Might be tight |

---

## 🧪 Testing Your Setup

### 1. Start Docker Services

```bash
cd D:\Content_2.0\rag_chatbot
docker-compose up -d
```

### 2. Check Health

```bash
# Wait 30 seconds for services to start, then:
curl http://localhost:8000/health
```

### 3. Index Sample Documents

```bash
# Copy sample data to container
docker cp data/sample rag_chatbot-rag-api-1:/app/data/

# Index it
docker exec rag_chatbot-rag-api-1 python scripts/index_documents.py /app/data/sample/
```

### 4. Test Query

```bash
# PowerShell
$body = @{
    query = "What is RAG?"
    top_k = 3
    include_sources = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/query -Method POST -Body $body -ContentType "application/json"
```

---

## 🐛 Troubleshooting

### Issue: "Connection refused" to Ollama

**Solution**:
```bash
# Make sure Ollama is running:
ollama serve

# Check if it's accessible:
curl http://localhost:11434/api/tags
```

### Issue: "Model not found"

**Solution**:
```bash
# Pull the model first:
ollama pull mistral

# Verify it's installed:
ollama list
```

### Issue: Out of VRAM

**Solution**:
```bash
# Use a smaller model:
ollama pull phi

# Update docker-compose.yml:
OLLAMA_MODEL=phi
```

### Issue: Slow responses

**Causes**:
- Model too large for VRAM → swapping to RAM
- First query is slow (model loading)

**Solutions**:
- Use smaller model (`phi` or `mistral`)
- Pre-load model: `ollama run mistral "test"`

---

## 💡 Performance Tips

### 1. Keep Ollama Running
Leave `ollama serve` running in the background for faster responses.

### 2. Pre-warm the Model
```bash
# Load model into VRAM before first query:
ollama run mistral "test"
```

### 3. Adjust Context Length
For faster responses with less VRAM:

```bash
# Create a Modelfile
echo "FROM mistral" > Modelfile
echo "PARAMETER num_ctx 2048" >> Modelfile
ollama create mistral-fast -f Modelfile

# Use in docker-compose.yml:
OLLAMA_MODEL=mistral-fast
```

---

## 🔄 Complete Workflow Summary

1. **Install Ollama** on your Windows laptop
2. **Pull model**: `ollama pull mistral`
3. **Start Ollama**: `ollama serve` (keep running)
4. **Start Docker**: `docker-compose up -d`
5. **Index data**: Via API or CLI
6. **Query**: Via API, CLI, or web interface

---

## 📝 What Happens Under the Hood

```
Your Query
    ↓
Docker Container (RAG API)
    ↓
Embedding Model (in container) → Vector Search → Retrieve Context
    ↓
Format Prompt with Context
    ↓
http://host.docker.internal:11434 (Bridge to your laptop)
    ↓
Ollama on Your Laptop (RTX 4050)
    ↓
Generate Response
    ↓
Return to API → Return to You
```

---

## 🎯 Next Steps

After setting up Ollama:

1. **Test Ollama**: `ollama run mistral "What is AI?"`
2. **Start RAG**: `docker-compose up -d`
3. **Index documents**: See testing section above
4. **Query**: Start using your RAG chatbot!

---

## 📞 Quick Reference

**Ollama Commands**:
```bash
ollama list                    # List installed models
ollama pull <model>           # Download a model
ollama run <model> "prompt"   # Test a model
ollama rm <model>             # Remove a model
ollama serve                  # Start Ollama service
```

**Docker Commands**:
```bash
docker-compose up -d          # Start services
docker-compose down           # Stop services
docker-compose logs rag-api   # View logs
docker-compose ps             # Check status
```

**RAG API Endpoints**:
- Health: `http://localhost:8000/health`
- Docs: `http://localhost:8000/docs`
- Query: `POST http://localhost:8000/query`
- Index: `POST http://localhost:8000/index`

---

**Your setup is optimized for local Ollama! Just install the model and you're ready to go! 🚀**

