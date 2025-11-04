"""CLI script for indexing documents."""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from src.adapters.embedding.local import LocalTextEmbeddingAdapter
from src.adapters.embedding.openai import OpenAIEmbeddingAdapter
from src.config.config import get_config
from src.extractors.base import ExtractorFactory
from src.utils.chunking import chunk_text
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.faiss import FAISSVectorStore
from src.vectorstore.memory import InMemoryVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_all_files(path: Path, recursive: bool = True):
    """Get all supported files from path.

    Args:
        path: Directory or file path
        recursive: Whether to search recursively

    Yields:
        File paths
    """
    if path.is_file():
        yield path
    elif path.is_dir():
        if recursive:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    yield file_path
        else:
            for file_path in path.glob("*"):
                if file_path.is_file():
                    yield file_path


def main():
    """Main indexing function."""
    parser = argparse.ArgumentParser(description="Index documents into vector store")
    parser.add_argument("path", type=str, help="Path to file or directory to index")
    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Recursively index directories"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    logger.info(f"Starting indexing with configuration:")
    logger.info(f"  Embedding provider: {config.embedding_provider}")
    logger.info(f"  Vector store provider: {config.vectorstore_provider}")
    logger.info(f"  Chunk size: {config.chunk_size}")
    logger.info(f"  Chunk overlap: {config.chunk_overlap}")

    # Initialize embedding adapter
    if config.embedding_provider.lower() == "local":
        embedding_adapter = LocalTextEmbeddingAdapter(
            model_name=config.embedding_model, device=config.embedding_device
        )
    elif config.embedding_provider.lower() == "openai":
        if not config.openai_api_key:
            logger.error("OPENAI_API_KEY not set")
            sys.exit(1)
        embedding_adapter = OpenAIEmbeddingAdapter(
            api_key=config.openai_api_key, model=config.embedding_model
        )
    else:
        logger.error(f"Unknown embedding provider: {config.embedding_provider}")
        sys.exit(1)

    # Initialize vector store
    embedding_dim = embedding_adapter.get_embedding_dimension()

    if config.vectorstore_provider.lower() == "chroma":
        vector_store = ChromaVectorStore(
            collection_name=config.vectorstore_collection_name,
            persist_directory=config.vectorstore_persist_path,
        )
    elif config.vectorstore_provider.lower() == "faiss":
        index_path = Path(config.vectorstore_persist_path) / "index.faiss"
        metadata_path = Path(config.vectorstore_persist_path) / "metadata.pkl"
        vector_store = FAISSVectorStore(
            embedding_dim=embedding_dim,
            index_path=str(index_path),
            metadata_path=str(metadata_path),
        )
    elif config.vectorstore_provider.lower() == "memory":
        vector_store = InMemoryVectorStore()
    else:
        logger.error(f"Unknown vector store provider: {config.vectorstore_provider}")
        sys.exit(1)

    # Initialize extractor
    extractor_factory = ExtractorFactory()

    # Get files to index
    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path does not exist: {args.path}")
        sys.exit(1)

    files = list(get_all_files(path, args.recursive))
    logger.info(f"Found {len(files)} files to process")

    # Process files
    total_chunks = 0
    processed_files = 0
    failed_files = 0

    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Check if extractor supports this file
            if not extractor_factory.get_extractor(str(file_path)):
                logger.debug(f"Skipping unsupported file: {file_path}")
                continue

            # Extract content
            extracted = extractor_factory.extract(str(file_path))

            # Chunk content
            chunks = chunk_text(
                extracted.content,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )

            if not chunks:
                logger.warning(f"No chunks extracted from {file_path}")
                continue

            # Create IDs and metadata
            doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

            metadatas = []
            for i in range(len(chunks)):
                meta = extracted.metadata.copy()
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

            total_chunks += len(chunks)
            processed_files += 1

            logger.debug(f"Indexed {file_path.name}: {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            failed_files += 1
            continue

    # Persist vector store
    vector_store.persist()

    # Summary
    logger.info("=" * 60)
    logger.info("Indexing completed!")
    logger.info(f"  Files processed: {processed_files}")
    logger.info(f"  Files failed: {failed_files}")
    logger.info(f"  Total chunks indexed: {total_chunks}")
    logger.info(f"  Total documents in store: {vector_store.count()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

