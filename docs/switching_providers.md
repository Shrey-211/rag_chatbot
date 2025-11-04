# Switching Providers

This guide explains how to switch between different LLM, embedding, and vector store providers.

## LLM Providers

### Ollama (Default)

**Pros**: Free, runs locally, privacy-preserving, no API costs
**Cons**: Requires local compute, slower than cloud APIs

**Setup**:
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Configure `.env`:
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

**Available Models**:
- `llama2` - General purpose
- `llama2:13b` - Better quality, slower
- `mistral` - Fast and capable
- `codellama` - Code-focused
- `phi` - Very small, fast

### OpenAI

**Pros**: State-of-the-art quality, fast, reliable
**Cons**: Costs money, requires API key, data sent to OpenAI

**Setup**:
1. Get API key: https://platform.openai.com/api-keys
2. Configure `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

**Available Models**:
- `gpt-3.5-turbo` - Fast, affordable
- `gpt-4` - Highest quality
- `gpt-4-turbo-preview` - Balance of speed and quality

**Cost Considerations**:
- GPT-3.5-turbo: ~$0.0015 per 1K tokens
- GPT-4: ~$0.03 per 1K tokens
- Monitor usage at https://platform.openai.com/usage

### Mock (Testing Only)

**Use Case**: Testing, development without real LLM

**Setup**:
```env
LLM_PROVIDER=mock
```

Returns predefined responses. Not for production.

## Embedding Providers

### Local (sentence-transformers)

**Pros**: Free, privacy-preserving, no API calls
**Cons**: Requires initial model download, uses local compute

**Setup**:
```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu  # or cuda
```

**Recommended Models**:
- `all-MiniLM-L6-v2` - Fast, 384 dims, good quality (default)
- `all-mpnet-base-v2` - Better quality, 768 dims, slower
- `multi-qa-mpnet-base-dot-v1` - Optimized for Q&A

**Performance**: First run downloads model (~80-400MB), then cached locally.

### OpenAI Embeddings

**Pros**: High quality, consistent, managed service
**Cons**: API costs, data sent to OpenAI

**Setup**:
```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=text-embedding-ada-002
```

**Available Models**:
- `text-embedding-ada-002` - 1536 dims, $0.0001 per 1K tokens
- `text-embedding-3-small` - 1536 dims, improved quality
- `text-embedding-3-large` - 3072 dims, best quality

**Important**: Once you index with one embedding model, you must use the same model for queries. Switching requires re-indexing all documents.

## Vector Store Providers

### ChromaDB (Default)

**Pros**: Easy setup, persistent, good for moderate scale
**Cons**: Slower than FAISS at very large scale

**Setup**:
```env
VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=./data/chroma
VECTORSTORE_COLLECTION_NAME=rag_documents
```

**Best For**: 10K - 1M documents, development, production with persistence

### FAISS

**Pros**: Very fast, memory efficient, scales to millions
**Cons**: Less feature-rich, requires manual persistence

**Setup**:
```env
VECTORSTORE_PROVIDER=faiss
VECTORSTORE_PERSIST_PATH=./data/faiss
```

**Best For**: Large-scale (1M+ documents), high-performance requirements

**Note**: FAISS stores index in memory. Call `persist()` to save.

### In-Memory

**Pros**: Simple, no dependencies
**Cons**: Not persistent, limited scale

**Setup**:
```env
VECTORSTORE_PROVIDER=memory
```

**Best For**: Testing, demos with small datasets

## Migration Scenarios

### Scenario 1: Switching LLM (Same Embeddings)

**From Ollama to OpenAI**:
1. Update `.env`: `LLM_PROVIDER=openai`
2. Add `OPENAI_API_KEY`
3. Restart server

✅ No re-indexing needed (embeddings unchanged)

### Scenario 2: Switching Embeddings

**From Local to OpenAI Embeddings**:
1. Clear vector store: `rm -rf ./data/chroma`
2. Update `.env`:
```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```
3. Re-index all documents: `python scripts/index_documents.py ./data/sample/`

⚠️ Re-indexing required (different embedding dimensions)

### Scenario 3: Switching Vector Store

**From Chroma to FAISS**:
1. Update `.env`: `VECTORSTORE_PROVIDER=faiss`
2. Re-index documents (different storage format)

⚠️ Re-indexing required (different storage backends)

## Configuration Templates

### Local Development (Free)

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=./data/chroma
```

### Production (High Quality)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large

VECTORSTORE_PROVIDER=chroma
VECTORSTORE_PERSIST_PATH=/data/chroma
```

### Production (Cost-Optimized)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

VECTORSTORE_PROVIDER=faiss
VECTORSTORE_PERSIST_PATH=/data/faiss
```

### High-Scale (1M+ docs)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-3.5-turbo

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

VECTORSTORE_PROVIDER=faiss
VECTORSTORE_PERSIST_PATH=/data/faiss
```

## Compatibility Matrix

| LLM | Embedding | Vector Store | Notes |
|-----|-----------|--------------|-------|
| Ollama | Local | Chroma | ✅ Default, fully local |
| Ollama | Local | FAISS | ✅ Best performance, local |
| Ollama | OpenAI | Chroma | ⚠️ Mixed (LLM local, embeddings cloud) |
| OpenAI | Local | Chroma | ⚠️ Mixed (LLM cloud, embeddings local) |
| OpenAI | OpenAI | Chroma | ✅ Production ready |
| OpenAI | OpenAI | FAISS | ✅ High performance |
| Mock | Local | Memory | ✅ Testing only |

## Cost Estimation

### Example: 1000 documents, 100 queries/day

**Fully Local (Ollama + Local Embeddings)**:
- Cost: $0/month
- Compute: Local CPU/GPU

**OpenAI LLM + Local Embeddings**:
- Embedding: $0 (local)
- LLM: ~$10-30/month (depends on response length)
- Total: ~$10-30/month

**Fully OpenAI**:
- Embedding (one-time): ~$0.10 for 1000 docs
- LLM: ~$10-30/month for queries
- Total: ~$10-30/month ongoing

## Performance Comparison

### Embedding Speed (1000 documents)

| Provider | Time | Cost |
|----------|------|------|
| Local (CPU) | ~30s | $0 |
| Local (GPU) | ~5s | $0 |
| OpenAI | ~10s | ~$0.10 |

### Query Speed (single query)

| LLM | Time | Cost |
|-----|------|------|
| Ollama (llama2) | ~5-10s | $0 |
| OpenAI (gpt-3.5) | ~2-3s | ~$0.002 |
| OpenAI (gpt-4) | ~5-8s | ~$0.02 |

### Vector Search Speed (100K documents)

| Store | Time |
|-------|------|
| Chroma | ~50ms |
| FAISS | ~10ms |
| Memory | ~200ms |

## Best Practices

1. **Start Local**: Use Ollama + local embeddings for development
2. **Match Use Case**: Choose based on scale, budget, privacy needs
3. **Monitor Costs**: Track OpenAI usage regularly
4. **Test Before Switching**: Evaluate quality before production migration
5. **Document Choices**: Keep notes on why each provider was chosen
6. **Version Lock**: Pin model versions in production
7. **Fallback Strategy**: Have backup provider configured

## Troubleshooting

### "Embedding dimension mismatch"

**Cause**: Changed embedding provider without re-indexing

**Fix**: Clear vector store and re-index with new provider

### "API key invalid"

**Cause**: Incorrect or expired OpenAI key

**Fix**: Verify key at https://platform.openai.com/api-keys

### "Model not found"

**Cause**: Ollama model not pulled locally

**Fix**: `ollama pull <model-name>`

