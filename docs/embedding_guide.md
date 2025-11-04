# Embedding Guide

## What are Embeddings?

Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings, enabling semantic search.

**Example**:
- "dog" → [0.2, 0.8, 0.1, ...]
- "puppy" → [0.3, 0.7, 0.2, ...] (similar to "dog")
- "car" → [0.9, 0.1, 0.3, ...] (different)

## Embedding Dimensions

Different models produce different dimensional embeddings:

| Model | Dimensions | Size | Quality |
|-------|------------|------|---------|
| all-MiniLM-L6-v2 | 384 | 80MB | Good ⭐⭐⭐ |
| all-mpnet-base-v2 | 768 | 420MB | Better ⭐⭐⭐⭐ |
| text-embedding-ada-002 | 1536 | API | Best ⭐⭐⭐⭐⭐ |
| text-embedding-3-large | 3072 | API | Best ⭐⭐⭐⭐⭐ |

**Trade-off**: Higher dimensions = better quality but more storage/compute

## Choosing an Embedding Model

### For Development

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**Why**: Fast, free, good enough for testing

### For Production (Privacy-Focused)

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-mpnet-base-v2
```

**Why**: Better quality, still local, no API costs

### For Production (Quality-Focused)

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
```

**Why**: State-of-the-art quality, managed service

### For Multi-Language

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

**Why**: Supports 50+ languages

## Embedding Workflow

### 1. Document Indexing

```
Document → Chunks → Embed Each Chunk → Store Vectors
```

**Example**:
```python
from src.adapters.embedding.local import LocalTextEmbeddingAdapter

adapter = LocalTextEmbeddingAdapter(model_name="all-MiniLM-L6-v2")
embeddings = adapter.embed_texts(["Hello world", "How are you"])
print(embeddings.shape)  # (2, 384)
```

### 2. Query Processing

```
Query → Embed Query → Search Similar Vectors → Return Documents
```

**Example**:
```python
query_embedding = adapter.embed_texts(["greeting"])[0]
# Use query_embedding to search vector store
```

## Advanced: Custom Embedding Models

### Using a Different Local Model

**Step 1**: Find a model on Hugging Face
- Browse: https://huggingface.co/models?library=sentence-transformers

**Step 2**: Update configuration
```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-MiniLM-L3-v2
```

**Step 3**: Re-index documents

### Domain-Specific Models

For specialized domains, consider fine-tuning:

1. **Legal**: `nlpaueb/legal-bert-base-uncased`
2. **Medical**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
3. **Code**: `microsoft/codebert-base`

## Image and Video Embeddings (Future)

The architecture supports media embeddings via `MediaEmbeddingAdapter` interface.

### Planned Support

**Image Embedding**:
```python
from src.adapters.embedding.media import ImageEmbeddingAdapter

adapter = ImageEmbeddingAdapter(model="clip-vit-base")
image_embedding = adapter.embed_image("photo.jpg")
```

**Video Embedding**:
```python
video_embedding = adapter.embed_video("video.mp4")
# Extracts frames and aggregates embeddings
```

### Where to Add Image/Video Support

**File**: `src/adapters/embedding/base.py`
```python
class MediaEmbeddingAdapter(ABC):
    @abstractmethod
    def embed_image(self, path: str) -> np.ndarray:
        """Embed image file."""
        pass
    
    @abstractmethod
    def embed_video(self, path: str) -> np.ndarray:
        """Embed video file."""
        pass
```

**Implementation** (create `src/adapters/embedding/media.py`):
```python
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbeddingAdapter(MediaEmbeddingAdapter):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def embed_image(self, path: str) -> np.ndarray:
        image = Image.open(path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().numpy()[0]
```

**Integration**:
1. Add to `src/extractors/` an `ImageExtractor`
2. Update `ExtractorFactory` to handle `.jpg`, `.png`
3. Store image embeddings in vector store
4. Query with text or image

## Performance Optimization

### Batch Processing

**Bad** (slow):
```python
for text in texts:
    embedding = adapter.embed_texts([text])
```

**Good** (fast):
```python
embeddings = adapter.embed_texts(texts)  # Batch all at once
```

### GPU Acceleration

```env
EMBEDDING_DEVICE=cuda
```

**Requirements**:
- NVIDIA GPU
- CUDA installed
- PyTorch with CUDA support

**Speed improvement**: 5-10x faster

### Caching

Embeddings are cached in vector store. Don't re-embed the same document.

## Troubleshooting

### "Model not found"

**Cause**: Model name incorrect or not downloaded

**Fix**:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")  # Downloads if needed
```

### "CUDA out of memory"

**Cause**: GPU memory exhausted

**Fix**:
1. Reduce batch size: `batch_size=16`
2. Use smaller model: `all-MiniLM-L6-v2`
3. Use CPU: `EMBEDDING_DEVICE=cpu`

### "Embedding dimension mismatch"

**Cause**: Vector store has different dimension than current model

**Fix**: Clear vector store and re-index

## Best Practices

1. **Consistency**: Use same model for indexing and querying
2. **Testing**: Evaluate quality on sample queries before production
3. **Versioning**: Pin model versions in production
4. **Monitoring**: Track embedding quality over time
5. **Storage**: Budget for vector storage (4 bytes × dims × num_chunks)

## Quality Evaluation

### Test Query Retrieval

```python
# After indexing, test retrievals
queries = [
    "What is machine learning?",
    "Explain neural networks",
    "How does backpropagation work?"
]

for query in queries:
    results = retriever.retrieve(query, top_k=3)
    print(f"\nQuery: {query}")
    for r in results:
        print(f"  Score: {r.score:.3f} - {r.content[:100]}")
```

### Compare Models

Index same corpus with different models and compare retrieval quality.

## Further Reading

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Model rankings

