"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: Optional[int] = Field(default=5, ge=1, le=100, description="Number of context documents to retrieve")
    include_sources: bool = Field(default=True, description="Include source documents in response")
    stream: bool = Field(default=False, description="Stream the response")


class SourceDocument(BaseModel):
    """Source document metadata."""

    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    sources: Optional[List[SourceDocument]] = None
    prompt: Optional[str] = None
    llm_metadata: Optional[Dict[str, Any]] = None


class IndexRequest(BaseModel):
    """Request model for indexing endpoint."""

    text: Optional[str] = Field(default=None, min_length=1, description="Text content to index")
    url: Optional[str] = Field(default=None, description="URL to fetch and index")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Document metadata")
    
    @model_validator(mode='after')
    def validate_text_or_url(self) -> 'IndexRequest':
        """Validate that either text or url is provided."""
        if not self.text and not self.url:
            raise ValueError("Either text or url must be provided")
        return self


class IndexResponse(BaseModel):
    """Response model for indexing endpoint."""

    success: bool
    document_id: str
    num_chunks: int
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    llm_healthy: bool
    embedding_healthy: bool
    vectorstore_count: int
    config: Dict[str, Any]


class StatsResponse(BaseModel):
    """Response model for statistics."""

    total_documents: int
    vectorstore_provider: str
    llm_provider: str
    embedding_provider: str
    embedding_dimension: int


class DocumentInfo(BaseModel):
    """Document information response."""

    document_id: str
    filename: Optional[str] = None
    content_type: Optional[str] = None
    source: Optional[str] = None
    num_chunks: int
    indexed_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class DocumentListResponse(BaseModel):
    """Response model for document list."""

    documents: List[DocumentInfo]
    total: int
    limit: int
    offset: int

