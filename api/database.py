"""Database module for storing document metadata."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, select, text
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
    file_path = Column(String(512), nullable=True)  # Path to stored file
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
        """Create tables if they don't exist and run migrations."""
        async with self.engine.begin() as conn:
            # Create tables if they don't exist
            await conn.run_sync(Base.metadata.create_all)
            
            # Run migrations to add missing columns
            await self._run_migrations(conn)
        
        logger.info(f"Document database initialized at {self.db_path}")
    
    async def _run_migrations(self, conn):
        """Run database migrations to add missing columns.
        
        Args:
            conn: SQLAlchemy connection
        """
        try:
            # Check if file_path column exists
            result = await conn.execute(text("PRAGMA table_info(documents)"))
            columns = [row[1] for row in result.fetchall()]
            
            # Add file_path column if it doesn't exist
            if "file_path" not in columns:
                logger.info("Adding file_path column to documents table...")
                await conn.execute(text(
                    "ALTER TABLE documents ADD COLUMN file_path VARCHAR(512)"
                ))
                logger.info("Migration: file_path column added successfully")
        except Exception as e:
            logger.error(f"Migration error: {e}")
            # Don't fail initialization if migration fails
            pass

    async def add_document(
        self,
        document_id: str,
        num_chunks: int,
        filename: Optional[str] = None,
        file_path: Optional[str] = None,
        content_type: Optional[str] = None,
        source: Optional[str] = None,
        metadata_json: Optional[str] = None,
    ) -> DocumentMetadata:
        """Add or update a document in the database (upsert).

        Args:
            document_id: Unique document identifier
            num_chunks: Number of chunks created
            filename: Original filename if uploaded
            file_path: Path to stored file on disk
            content_type: MIME type
            source: Source description
            metadata_json: JSON string of additional metadata

        Returns:
            Created or updated DocumentMetadata object
        """
        async with self.async_session() as session:
            async with session.begin():
                # Check if document already exists
                stmt = select(DocumentMetadata).where(DocumentMetadata.document_id == document_id)
                result = await session.execute(stmt)
                existing_doc = result.scalar_one_or_none()
                
                if existing_doc:
                    # Update existing document
                    logger.info(f"Updating existing document: {document_id}")
                    existing_doc.filename = filename
                    existing_doc.file_path = file_path
                    existing_doc.content_type = content_type
                    existing_doc.source = source
                    existing_doc.num_chunks = num_chunks
                    existing_doc.metadata_json = metadata_json
                    existing_doc.indexed_at = datetime.utcnow()
                    doc = existing_doc
                else:
                    # Create new document
                    logger.info(f"Creating new document: {document_id}")
                    doc = DocumentMetadata(
                        document_id=document_id,
                        filename=filename,
                        file_path=file_path,
                        content_type=content_type,
                        source=source,
                        num_chunks=num_chunks,
                        metadata_json=metadata_json,
                    )
                    session.add(doc)
                # Commit is handled automatically by session.begin() context manager
            # With expire_on_commit=False, object remains accessible after session closes
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
            doc = result.scalar_one_or_none()
            # With expire_on_commit=False, object remains accessible after session closes
            return doc

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
            documents = list(result.scalars().all())
            # With expire_on_commit=False, objects remain accessible after session closes
            return documents

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

