"""FastAPI main application."""

import hashlib
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from api.database import DocumentDatabase
from api.models import (
    DocumentInfo,
    DocumentListResponse,
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
doc_database: Optional[DocumentDatabase] = None


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
    global llm_adapter, embedding_adapter, vector_store, retriever, extractor_factory, doc_database

    logger.info("Starting RAG application...")

    config = get_config()

    # Initialize document database
    doc_database = DocumentDatabase()
    await doc_database.initialize()

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
    if doc_database:
        await doc_database.close()
    # Close HTTP clients if they have close methods
    if llm_adapter and hasattr(llm_adapter, 'client') and hasattr(llm_adapter.client, 'close'):
        try:
            llm_adapter.client.close()
        except Exception as e:
            logger.warning(f"Error closing LLM adapter client: {e}")


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

    # Validate query
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

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

        # Build response - group chunks by document
        sources = None
        if request.include_sources and results:
            # Group chunks by document_id
            document_chunks = {}
            for r in results:
                doc_id = r.metadata.get("document_id") if r.metadata else None
                if not doc_id:
                    # Fallback: try to extract from chunk ID
                    if "_" in r.id:
                        doc_id = r.id.split("_")[0]
                    else:
                        doc_id = r.id
                
                if doc_id not in document_chunks:
                    document_chunks[doc_id] = {
                        "chunks": [],
                        "scores": [],
                        "metadata": r.metadata or {}
                    }
                
                document_chunks[doc_id]["chunks"].append({
                    "id": r.id,
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": r.score
                })
                document_chunks[doc_id]["scores"].append(r.score)
            
            # Fetch document metadata from database
            sources = []
            for doc_id, doc_data in document_chunks.items():
                # Get document info from database
                doc_metadata = None
                filename = None
                content_type = None
                has_file = False
                
                if doc_database:
                    try:
                        doc_metadata = await doc_database.get_document(doc_id)
                        if doc_metadata:
                            # Extract values immediately while object is valid
                            filename = doc_metadata.filename
                            content_type = doc_metadata.content_type
                            file_path_str = doc_metadata.file_path
                            # Check file existence using the string path
                            has_file = file_path_str is not None and Path(file_path_str).exists() if file_path_str else False
                    except Exception as e:
                        logger.warning(f"Could not fetch document {doc_id}: {e}")
                
                # Fallback to metadata if database lookup failed
                if not filename and doc_data["metadata"].get("filename"):
                    filename = doc_data["metadata"]["filename"]
                
                # Calculate best score
                best_score = max(doc_data["scores"]) if doc_data["scores"] else 0.0
                
                sources.append(SourceDocument(
                    document_id=doc_id,
                    filename=filename,
                    content_type=content_type,
                    score=best_score,
                    num_chunks=len(doc_data["chunks"]),
                    chunks=doc_data["chunks"][:3],  # Show up to 3 sample chunks
                    has_file=has_file
                ))
            
            # Sort by score descending
            sources.sort(key=lambda x: x.score, reverse=True)

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
            # Validate URL
            if not request.url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail="Invalid URL scheme. Must be http:// or https://")
            
            # Fetch URL content
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                try:
                    response = await client.get(request.url)
                    response.raise_for_status()
                    content = response.text
                except httpx.HTTPError as e:
                    raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
            doc_id = hashlib.md5(request.url.encode()).hexdigest()
        else:
            # This should not happen due to model validation, but keep as safety check
            raise HTTPException(status_code=400, detail="Either text or url must be provided")

        # Validate content
        if not content or not content.strip():
            raise HTTPException(status_code=400, detail="Content is empty")

        # Chunk content
        chunks = chunk_text(
            content, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from content")

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

        # Save to document database
        await doc_database.add_document(
            document_id=doc_id,
            num_chunks=len(chunks),
            source=request.metadata.get("source") if request.metadata else None,
            metadata_json=json.dumps(request.metadata) if request.metadata else None,
        )

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
        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Sanitize filename to prevent path traversal
        safe_filename = Path(file.filename).name
        if not safe_filename or safe_filename in (".", ".."):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Save uploaded file permanently
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename with document ID
        doc_id = hashlib.md5(safe_filename.encode()).hexdigest()
        file_extension = Path(safe_filename).suffix
        stored_filename = f"{doc_id}{file_extension}"
        file_path = upload_dir / stored_filename
        
        # Save file content
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)

        # Extract content for indexing
        extracted = extractor_factory.extract(str(file_path))

        # Validate extracted content
        if not extracted.content or not extracted.content.strip():
            raise HTTPException(status_code=400, detail="File contains no extractable text content")

        # Chunk content
        chunks = chunk_text(
            extracted.content,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from file content")

        # Create chunk IDs and metadata
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        metadatas = []
        for i in range(len(chunks)):
            meta = extracted.metadata.copy() if extracted.metadata else {}
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

        # Save to document database with file path (upsert)
        doc_record = await doc_database.add_document(
            document_id=doc_id,
            num_chunks=len(chunks),
            filename=safe_filename,
            file_path=str(file_path),
            content_type=file.content_type,
            metadata_json=json.dumps(extracted.metadata) if extracted.metadata else None,
        )

        return IndexResponse(
            success=True,
            document_id=doc_id,
            num_chunks=len(chunks),
            message=f"Successfully indexed file: {safe_filename} ({len(chunks)} chunks)",
        )

    except Exception as e:
        logger.error(f"File index error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List all indexed documents with metadata.
    
    Args:
        limit: Maximum number of documents to return (1-1000)
        offset: Number of documents to skip
    
    Returns:
        List of documents with metadata
    """
    if not doc_database:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        documents = await doc_database.list_documents(limit=limit, offset=offset)
        total = await doc_database.count_documents()

        doc_list = []
        for doc in documents:
            # Extract file_path string before checking existence
            file_path_str = doc.file_path
            has_file = file_path_str is not None and Path(file_path_str).exists() if file_path_str else False
            
            doc_list.append(DocumentInfo(
                document_id=doc.document_id,
                filename=doc.filename,
                file_path=file_path_str,
                content_type=doc.content_type,
                source=doc.source,
                num_chunks=doc.num_chunks,
                indexed_at=doc.indexed_at,
                metadata=json.loads(doc.metadata_json) if doc.metadata_json else None,
                has_file=has_file,
            ))

        return DocumentListResponse(
            documents=doc_list,
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}/file")
async def download_document(document_id: str):
    """View an uploaded document file in the browser.
    
    Args:
        document_id: Document identifier
        
    Returns:
        File response with the document displayed inline
    """
    if not doc_database:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        doc = await doc_database.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not doc.file_path:
            raise HTTPException(status_code=404, detail="File not available for this document")
        
        file_path = Path(doc.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Read file content
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        # Determine media type based on file extension if not set
        media_type = doc.content_type
        if not media_type:
            suffix = file_path.suffix.lower()
            # Map common file extensions to media types that browsers can display
            media_type_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".html": "text/html",
                ".htm": "text/html",
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
                ".json": "application/json",
                ".xml": "application/xml",
                ".csv": "text/csv",
            }
            media_type = media_type_map.get(suffix, "application/octet-stream")
        
        # Set Content-Disposition to inline so browser displays instead of downloads
        # Browsers will display text, images, PDFs, etc. inline when possible
        headers = {
            "Content-Disposition": f"inline; filename=\"{doc.filename or 'document'}\""
        }
        
        return Response(
            content=file_content,
            media_type=media_type,
            headers=headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"View document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.api_reload,
    )

