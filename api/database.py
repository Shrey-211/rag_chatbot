"""Database module for storing document metadata."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


class DocumentMetadata(Base):
    """Document metadata table."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=True)
    content_type = Column(String(100), nullable=True)
    source = Column(String(255), nullable=True)
    num_chunks = Column(Integer, nullable=False)
    metadata_json = Column(Text, nullable=True)
    indexed_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentDatabase:
    """Database manager for document metadata."""

    def __init__(self, db_path: str = "data/documents.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Create async engine
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
            future=True,
        )

        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Document database initialized at {self.db_path}")

    async def add_document(
        self,
        document_id: str,
        num_chunks: int,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        source: Optional[str] = None,
        metadata_json: Optional[str] = None,
    ) -> DocumentMetadata:
        """Add a new document to the database.

        Args:
            document_id: Unique document identifier
            num_chunks: Number of chunks created
            filename: Original filename if uploaded
            content_type: MIME type
            source: Source description
            metadata_json: JSON string of additional metadata

        Returns:
            Created DocumentMetadata object
        """
        async with self.async_session() as session:
            async with session.begin():
                doc = DocumentMetadata(
                    document_id=document_id,
                    filename=filename,
                    content_type=content_type,
                    source=source,
                    num_chunks=num_chunks,
                    metadata_json=metadata_json,
                )
                session.add(doc)
                # Commit is handled automatically by session.begin() context manager
                await session.refresh(doc)
                return doc

    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document by ID.

        Args:
            document_id: Document identifier

        Returns:
            DocumentMetadata if found, None otherwise
        """
        async with self.async_session() as session:
            stmt = select(DocumentMetadata).where(DocumentMetadata.document_id == document_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_documents(
        self, limit: int = 100, offset: int = 0
    ) -> List[DocumentMetadata]:
        """List all documents ordered by indexed_at DESC.

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of DocumentMetadata objects
        """
        async with self.async_session() as session:
            from sqlalchemy import select

            stmt = (
                select(DocumentMetadata)
                .order_by(DocumentMetadata.indexed_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def count_documents(self) -> int:
        """Count total documents in database.

        Returns:
            Total document count
        """
        async with self.async_session() as session:
            from sqlalchemy import func, select

            stmt = select(func.count()).select_from(DocumentMetadata)
            result = await session.execute(stmt)
            return result.scalar() or 0

    async def delete_document(self, document_id: str) -> bool:
        """Delete document by ID.

        Args:
            document_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        async with self.async_session() as session:
            async with session.begin():
                stmt = select(DocumentMetadata).where(DocumentMetadata.document_id == document_id)
                result = await session.execute(stmt)
                doc = result.scalar_one_or_none()
                if doc:
                    await session.delete(doc)
                    # Commit is handled automatically by session.begin() context manager
                    return True
                return False

    async def close(self):
        """Close database connection."""
        await self.engine.dispose()

