"""FastAPI main application."""

import hashlib
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.models import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    StatsResponse,
)
from src.adapters.embedding.base import EmbeddingAdapter
from src.adapters.embedding.local import LocalTextEmbeddingAdapter
from src.adapters.embedding.openai import OpenAIEmbeddingAdapter
from src.adapters.llm.base import LLMAdapter
from src.adapters.llm.mock import MockLLMAdapter
from src.adapters.llm.ollama import OllamaAdapter
from src.adapters.llm.openai import OpenAIAdapter
from src.config.config import get_config, get_yaml_config
from src.extractors.base import ExtractorFactory
from src.retriever.retriever import Retriever
from src.utils.chunking import chunk_text
from src.utils.prompts import RAG_WITH_SYSTEM
from src.vectorstore.base import VectorStore
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.faiss import FAISSVectorStore
from src.vectorstore.memory import InMemoryVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
llm_adapter: Optional[LLMAdapter] = None
embedding_adapter: Optional[EmbeddingAdapter] = None
vector_store: Optional[VectorStore] = None
retriever: Optional[Retriever] = None
extractor_factory: Optional[ExtractorFactory] = None


def create_llm_adapter(config) -> LLMAdapter:
    """Create LLM adapter based on configuration."""
    provider = config.llm_provider.lower()

    if provider == "ollama":
        return OllamaAdapter(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
        )
    elif provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIAdapter(
            api_key=config.openai_api_key,
            model=config.openai_model,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
        )
    elif provider == "mock":
        return MockLLMAdapter()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def create_embedding_adapter(config) -> EmbeddingAdapter:
    """Create embedding adapter based on configuration."""
    provider = config.embedding_provider.lower()

    if provider == "local":
        return LocalTextEmbeddingAdapter(
            model_name=config.embedding_model,
            device=config.embedding_device,
        )
    elif provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAIEmbeddingAdapter(
            api_key=config.openai_api_key,
            model=config.embedding_model,
            timeout=config.request_timeout,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def create_vector_store(config, embedding_dim: int) -> VectorStore:
    """Create vector store based on configuration."""
    provider = config.vectorstore_provider.lower()

    if provider == "chroma":
        return ChromaVectorStore(
            collection_name=config.vectorstore_collection_name,
            persist_directory=config.vectorstore_persist_path,
        )
    elif provider == "faiss":
        index_path = Path(config.vectorstore_persist_path) / "index.faiss"
        metadata_path = Path(config.vectorstore_persist_path) / "metadata.pkl"
        return FAISSVectorStore(
            embedding_dim=embedding_dim,
            index_path=str(index_path),
            metadata_path=str(metadata_path),
        )
    elif provider == "memory":
        return InMemoryVectorStore()
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global llm_adapter, embedding_adapter, vector_store, retriever, extractor_factory

    logger.info("Starting RAG application...")

    config = get_config()

    # Initialize adapters
    llm_adapter = create_llm_adapter(config)
    embedding_adapter = create_embedding_adapter(config)

    # Initialize vector store
    embedding_dim = embedding_adapter.get_embedding_dimension()
    vector_store = create_vector_store(config, embedding_dim)

    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedding_adapter=embedding_adapter,
        top_k=config.top_k_results,
    )

    # Initialize extractor factory
    extractor_factory = ExtractorFactory()

    logger.info("RAG application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down RAG application...")
    if vector_store:
        vector_store.persist()


# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Production-quality RAG pipeline with modular adapters",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
config = get_config()
yaml_config = get_yaml_config()
cors_origins = yaml_config.get("api", {}).get("cors_origins", ["*"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    llm_healthy = llm_adapter.health_check() if llm_adapter else False
    embedding_healthy = embedding_adapter.health_check() if embedding_adapter else False
    vectorstore_count = vector_store.count() if vector_store else 0

    return HealthResponse(
        status="healthy" if llm_healthy and embedding_healthy else "degraded",
        llm_healthy=llm_healthy,
        embedding_healthy=embedding_healthy,
        vectorstore_count=vectorstore_count,
        config={
            "llm_provider": config.llm_provider,
            "embedding_provider": config.embedding_provider,
            "vectorstore_provider": config.vectorstore_provider,
        },
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    stats = retriever.get_statistics() if retriever else {}

    return StatsResponse(
        total_documents=stats.get("total_documents", 0),
        vectorstore_provider=config.vectorstore_provider,
        llm_provider=config.llm_provider,
        embedding_provider=config.embedding_provider,
        embedding_dimension=stats.get("embedding_dim", 0),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint for RAG."""
    if not retriever or not llm_adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Retrieve relevant documents
        results = retriever.retrieve(query=request.query, top_k=request.top_k)

        # Format context
        context = retriever.format_context(results, include_metadata=False)

        # Build prompt
        yaml_cfg = get_yaml_config()
        prompt_template = yaml_cfg.get("prompts", {}).get("rag_template")
        if prompt_template:
            from src.utils.prompts import PromptTemplate

            template = PromptTemplate(prompt_template)
            prompt = template.format(query=request.query, context=context)
        else:
            prompt = RAG_WITH_SYSTEM.format(query=request.query, context=context)

        # Generate answer
        if request.stream:
            # TODO: Implement streaming response
            raise HTTPException(status_code=501, detail="Streaming not yet implemented")

        response = llm_adapter.generate(prompt)

        # Build response
        sources = None
        if request.include_sources:
            sources = [
                SourceDocument(
                    id=r.id, content=r.content, metadata=r.metadata, score=r.score
                )
                for r in results
            ]

        return QueryResponse(
            answer=response.text,
            sources=sources,
            prompt=prompt if request.include_sources else None,
            llm_metadata={
                "model": response.model,
                "usage": response.usage,
            },
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_document(request: IndexRequest):
    """Index a document from text or URL."""
    if not vector_store or not embedding_adapter:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Get content
        if request.text:
            content = request.text
            doc_id = hashlib.md5(content.encode()).hexdigest()
        elif request.url:
            # Fetch URL content
            async with httpx.AsyncClient() as client:
                response = await client.get(request.url)
                response.raise_for_status()
                content = response.text
            doc_id = hashlib.md5(request.url.encode()).hexdigest()
        else:
            raise HTTPException(status_code=400, detail="Either text or url must be provided")

        # Chunk content
        chunks = chunk_text(
            content, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

        # Create IDs and metadata
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = []
        for i in range(len(chunks)):
            meta = request.metadata.copy() if request.metadata else {}
            meta["chunk_index"] = i
            meta["total_chunks"] = len(chunks)
            meta["document_id"] = doc_id
            metadatas.append(meta)

        # Embed chunks
        embeddings = embedding_adapter.embed_texts(chunks)

        # Upsert to vector store
        vector_store.upsert(
            ids=chunk_ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )

        vector_store.persist()

        return IndexResponse(
            success=True,
            document_id=doc_id,
            num_chunks=len(chunks),
            message=f"Successfully indexed {len(chunks)} chunks",
        )

    except Exception as e:
        logger.error(f"Index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/file", response_model=IndexResponse)
async def index_file(file: UploadFile = File(...)):
    """Index a document from uploaded file."""
    if not vector_store or not embedding_adapter or not extractor_factory:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Save uploaded file temporarily
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract content
        try:
            extracted = extractor_factory.extract(str(file_path))
        finally:
            # Clean up temp file
            file_path.unlink(missing_ok=True)

        # Chunk content
        chunks = chunk_text(
            extracted.content,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        # Create IDs and metadata
        doc_id = hashlib.md5(file.filename.encode()).hexdigest()
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        metadatas = []
        for i in range(len(chunks)):
            meta = extracted.metadata.copy()
            meta["chunk_index"] = i
            meta["total_chunks"] = len(chunks)
            meta["document_id"] = doc_id
            metadatas.append(meta)

        # Embed and upsert
        embeddings = embedding_adapter.embed_texts(chunks)
        vector_store.upsert(
            ids=chunk_ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )

        vector_store.persist()

        return IndexResponse(
            success=True,
            document_id=doc_id,
            num_chunks=len(chunks),
            message=f"Successfully indexed file: {file.filename}",
        )

    except Exception as e:
        logger.error(f"File index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
    )

