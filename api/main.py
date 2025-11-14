"""FastAPI main application."""

import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request
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
from src.utils.dependency_checker import DependencyChecker
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
vision_adapter: Optional["VisionAdapter"] = None
personal_info_extractor: Optional["PersonalInfoExtractor"] = None


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
    global vision_adapter, personal_info_extractor

    logger.info("Starting RAG application...")

    config = get_config()
    
    # Check OCR dependencies
    logger.info("")
    DependencyChecker.report_status(verbose=True)
    logger.info("")

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
        min_score=config.min_relevance_score,
    )

    # Initialize extractor factory with OCR configuration
    ocr_config = yaml_config.get("ocr", {})
    extractor_factory = ExtractorFactory(ocr_config=ocr_config)

    # Initialize vision adapter and personal info extractor
    vision_config = yaml_config.get("llm", {}).get("vision", {})
    if vision_config.get("enabled", False):
        try:
            from src.adapters.vision.ollama import OllamaVisionAdapter
            from src.services.personal_info_extractor import PersonalInfoExtractor
            
            vision_adapter = OllamaVisionAdapter(
                base_url=vision_config.get("base_url", "http://localhost:11434"),
                model=vision_config.get("model", "llama3.2-vision:11b"),
                temperature=vision_config.get("temperature", 0.1),
                max_tokens=vision_config.get("max_tokens", 1024),
            )
            
            # Check if vision model is available
            if vision_adapter.health_check():
                logger.info(f"✓ Vision adapter initialized: {vision_config.get('model')}")
                
                personal_info_extractor = PersonalInfoExtractor(
                    vision_adapter=vision_adapter,
                    poppler_path=ocr_config.get("poppler_path"),
                    dpi=200,  # Lower DPI for faster processing
                )
                logger.info("✓ Personal information extractor initialized")
            else:
                logger.warning("⚠ Vision model not available. Personal info extraction disabled.")
                logger.warning(f"   Run: ollama pull {vision_config.get('model')}")
                vision_adapter = None
                personal_info_extractor = None
        except Exception as e:
            logger.warning(f"⚠ Could not initialize vision adapter: {e}")
            vision_adapter = None
            personal_info_extractor = None
    else:
        logger.info("Vision-based personal info extraction is disabled in config")

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


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming HTTP requests and their duration."""
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration_ms = (time.time() - start_time) * 1000
    status_emoji = "✓" if response.status_code < 400 else "✗"
    logger.info(f"← {status_emoji} {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration_ms:.0f}ms")
    
    return response


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

    logger.info(f"🔍 Received query: '{request.query[:100]}...' (top_k={request.top_k})")
    
    try:
        # Retrieve relevant documents
        logger.info(f"   Searching vector store for relevant documents...")
        results = retriever.retrieve(query=request.query, top_k=request.top_k)
        
        # Check if we got any relevant results
        if not results:
            logger.warning(f"   ⚠ No relevant documents found for query")
            return QueryResponse(
                answer="I couldn't find any relevant information in the knowledge base to answer your question. This could mean:\n\n"
                       "1. The information isn't in the indexed documents\n"
                       "2. The question is too different from the document content\n"
                       "3. The documents need to be re-indexed with better settings\n\n"
                       "Try rephrasing your question or check if relevant documents are indexed.",
                sources=[],
                llm_metadata={
                    "model": config.llm_provider,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
            )
        
        logger.info(f"   ✓ Found {len(results)} relevant chunks")

        # Enrich context with personal information
        enriched_results = []
        for result in results:
            # Get document_id from result
            doc_id = result.metadata.get("document_id") if result.metadata else None
            
            # Fetch personal info for this document
            personal_info_text = ""
            if doc_id and doc_database:
                try:
                    personal_info_records = await doc_database.get_personal_info(doc_id)
                    if personal_info_records:
                        # Format personal info as text to include in context
                        info_lines = ["\n--- Extracted Personal Information from this document ---"]
                        for record in personal_info_records:
                            entity_label = record.entity_type.replace('_', ' ').title()
                            info_lines.append(f"{entity_label}: {record.entity_value}")
                        personal_info_text = "\n".join(info_lines)
                except Exception as e:
                    logger.warning(f"Could not fetch personal info for context: {e}")
            
            # Create enriched result with personal info appended
            enriched_content = result.content
            if personal_info_text:
                enriched_content = result.content + personal_info_text
            
            # Create new result with enriched content
            from src.vectorstore.base import SearchResult
            enriched_result = SearchResult(
                id=result.id,
                content=enriched_content,
                metadata=result.metadata,
                score=result.score
            )
            enriched_results.append(enriched_result)
        
        # Format context with enriched results
        context = retriever.format_context(enriched_results, include_metadata=False)

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

        logger.info(f"   Generating answer using {config.llm_provider}...")
        response = llm_adapter.generate(prompt)
        logger.info(f"   ✓ Answer generated ({len(response.text)} chars)")

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
                personal_info_entities = None
                
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
                        
                        # Get personal information for this document
                        personal_info_records = await doc_database.get_personal_info(doc_id)
                        if personal_info_records:
                            from api.models import PersonalInfoEntity
                            personal_info_entities = [
                                PersonalInfoEntity(
                                    entity_type=record.entity_type,
                                    entity_value=record.entity_value,
                                    confidence=record.confidence,
                                    context=record.context,
                                    extracted_at=record.extracted_at,
                                )
                                for record in personal_info_records
                            ]
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
                    has_file=has_file,
                    personal_info=personal_info_entities,
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

    logger.info(f"📄 Indexing document from {'text' if request.text else 'URL'}...")
    
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

    logger.info(f"📁 Received file upload: {file.filename} ({file.content_type})")
    
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
        file_size_kb = len(file_content) / 1024
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info(f"   ✓ Saved to: {file_path} ({file_size_kb:.2f} KB)")

        # Extract content for indexing
        logger.info(f"   Extracting text content from {file_extension} file...")
        extracted = extractor_factory.extract(str(file_path))
        logger.info(f"   ✓ Extracted {len(extracted.content)} characters")

        # Validate extracted content
        if not extracted.content or not extracted.content.strip():
            logger.error(f"   ✗ No text content extracted from file")
            raise HTTPException(status_code=400, detail="File contains no extractable text content")

        # Chunk content
        logger.info(f"   Chunking text content...")
        chunks = chunk_text(
            extracted.content,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        if not chunks:
            logger.error(f"   ✗ No valid chunks created")
            raise HTTPException(status_code=400, detail="No valid chunks created from file content")

        # Create chunk IDs and metadata
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        metadatas = []
        for i in range(len(chunks)):
            meta = extracted.metadata.copy() if extracted.metadata else {}
            meta["chunk_index"] = i
            meta["total_chunks"] = len(chunks)
            meta["document_id"] = doc_id
            meta["filename"] = safe_filename
            metadatas.append(meta)

        # Embed and upsert
        embeddings = embedding_adapter.embed_texts(chunks)
        vector_store.upsert(
            ids=chunk_ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )

        logger.info(f"   Persisting vector store to disk...")
        vector_store.persist()

        # Note: We'll update the database record after vision analysis to include summary

        # Extract personal information using vision model (if enabled)
        personal_info_count = 0
        vision_performed = False
        document_summary = None
        
        if personal_info_extractor:
            try:
                logger.info(f"   🔍 Analyzing document with vision model...")
                entities, raw_response, summary = personal_info_extractor.extract_from_document(str(file_path))
                
                # Store document summary
                if summary:
                    document_summary = summary
                    logger.info(f"   ✓ Generated document summary")
                    
                    # Index the summary as a special chunk so it's searchable
                    summary_chunk_id = f"{doc_id}_summary"
                    summary_metadata = {
                        "document_id": doc_id,
                        "chunk_index": -1,  # Special marker for summary
                        "total_chunks": len(chunks) + 1,
                        "filename": safe_filename,
                        "is_document_summary": True,
                    }
                    
                    # Embed and store summary
                    summary_embedding = embedding_adapter.embed_texts([summary])
                    vector_store.upsert(
                        ids=[summary_chunk_id],
                        embeddings=summary_embedding,
                        documents=[summary],
                        metadatas=[summary_metadata]
                    )
                    logger.info(f"   ✓ Indexed document summary for semantic search")
                
                # Store extracted personal information
                if entities:
                    for entity in entities:
                        await doc_database.add_personal_info(
                            document_id=doc_id,
                            entity_type=entity['entity_type'],
                            entity_value=entity['entity_value'],
                            confidence=entity.get('confidence'),
                            context=entity.get('context'),
                            raw_extraction=raw_response[:1000] if raw_response else None,
                        )
                    
                    personal_info_count = len(entities)
                    logger.info(f"   ✓ Extracted and stored {personal_info_count} personal information entities")
                else:
                    logger.info(f"   ℹ No personal information entities extracted")
                
                vision_performed = True
                    
            except Exception as e:
                logger.warning(f"   ⚠ Vision analysis failed: {e}")
                # Don't fail the entire indexing if vision extraction fails
        
        # Save to document database with all metadata including summary
        logger.info(f"   Saving metadata to document database...")
        doc_record = await doc_database.add_document(
            document_id=doc_id,
            num_chunks=len(chunks) + (1 if document_summary else 0),  # Include summary chunk
            filename=safe_filename,
            file_path=str(file_path),
            content_type=file.content_type,
            metadata_json=json.dumps(extracted.metadata) if extracted.metadata else None,
            document_summary=document_summary,
        )

        logger.info(f"✅ Successfully indexed file: {safe_filename} (doc_id={doc_id}, {len(chunks)} chunks)")
        
        return IndexResponse(
            success=True,
            document_id=doc_id,
            num_chunks=len(chunks),
            message=f"Successfully indexed file: {safe_filename} ({len(chunks)} chunks)",
            personal_info_extracted=personal_info_count if vision_performed else None,
            vision_analysis_performed=vision_performed,
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
            
            # Get personal information for this document
            personal_info_entities = None
            try:
                personal_info_records = await doc_database.get_personal_info(doc.document_id)
                if personal_info_records:
                    from api.models import PersonalInfoEntity
                    personal_info_entities = [
                        PersonalInfoEntity(
                            entity_type=record.entity_type,
                            entity_value=record.entity_value,
                            confidence=record.confidence,
                            context=record.context,
                            extracted_at=record.extracted_at,
                        )
                        for record in personal_info_records
                    ]
            except Exception as e:
                logger.warning(f"Could not fetch personal info for {doc.document_id}: {e}")
            
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
                personal_info=personal_info_entities,
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


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its associated data.
    
    This endpoint deletes:
    1. Document metadata from the database
    2. Uploaded file from disk (if exists)
    3. All chunks/embeddings from the vector store
    
    Args:
        document_id: Document identifier
        
    Returns:
        Success message with deletion details
    """
    if not doc_database or not vector_store:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    logger.info(f"🗑️  Deleting document: {document_id}")
    
    try:
        # Step 1: Get document metadata (to find file path and num_chunks)
        doc = await doc_database.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        num_chunks = doc.num_chunks
        file_path = doc.file_path
        filename = doc.filename or document_id
        
        # Step 2: Delete file from disk if it exists
        file_deleted = False
        if file_path:
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                try:
                    file_path_obj.unlink()
                    logger.info(f"   ✓ Deleted file from disk: {file_path}")
                    file_deleted = True
                except Exception as e:
                    logger.warning(f"   ⚠ Could not delete file: {e}")
        
        # Step 3: Delete all chunks from vector store (including summary)
        # Generate all chunk IDs for this document
        chunk_ids = [f"{document_id}_{i}" for i in range(num_chunks)]
        chunk_ids.append(f"{document_id}_summary")  # Also delete summary chunk
        
        try:
            vector_store.delete(ids=chunk_ids)
            logger.info(f"   ✓ Deleted {len(chunk_ids)} chunks from vector store")
        except Exception as e:
            logger.error(f"   ✗ Error deleting chunks from vector store: {e}")
            # Continue with database deletion even if vector store fails
        
        # Step 4: Delete personal information from database
        personal_info_deleted = 0
        try:
            personal_info_deleted = await doc_database.delete_personal_info(document_id)
            if personal_info_deleted > 0:
                logger.info(f"   ✓ Deleted {personal_info_deleted} personal info entities")
        except Exception as e:
            logger.warning(f"   ⚠ Could not delete personal info: {e}")
        
        # Step 5: Delete document from database
        db_deleted = await doc_database.delete_document(document_id)
        if not db_deleted:
            raise HTTPException(status_code=404, detail="Document not found in database")
        
        logger.info(f"   ✓ Deleted document from database")
        
        # Step 6: Persist vector store changes
        try:
            vector_store.persist()
            logger.info(f"   ✓ Persisted vector store changes")
        except Exception as e:
            logger.warning(f"   ⚠ Could not persist vector store: {e}")
        
        logger.info(f"✅ Successfully deleted document: {filename}")
        
        return {
            "success": True,
            "document_id": document_id,
            "message": f"Successfully deleted document: {filename}",
            "details": {
                "database_deleted": True,
                "file_deleted": file_deleted,
                "chunks_deleted": len(chunk_ids),
                "personal_info_deleted": personal_info_deleted,
                "filename": filename
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
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

