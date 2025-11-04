"""CLI script for querying the RAG system."""

import argparse
import logging
import sys

from src.adapters.embedding.local import LocalTextEmbeddingAdapter
from src.adapters.embedding.openai import OpenAIEmbeddingAdapter
from src.adapters.llm.mock import MockLLMAdapter
from src.adapters.llm.ollama import OllamaAdapter
from src.adapters.llm.openai import OpenAIAdapter
from src.config.config import get_config
from src.retriever.retriever import Retriever
from src.utils.prompts import RAG_WITH_SYSTEM
from src.vectorstore.chroma import ChromaVectorStore
from src.vectorstore.faiss import FAISSVectorStore
from src.vectorstore.memory import InMemoryVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main query function."""
    parser = argparse.ArgumentParser(description="Query RAG system")
    parser.add_argument("query", type=str, nargs="?", help="Query text (or use interactive mode)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--top-k", type=int, default=None, help="Number of results to retrieve")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive query mode"
    )
    parser.add_argument(
        "--show-sources", action="store_true", help="Show source documents in output"
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    logger.info("Initializing RAG system...")

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
        from pathlib import Path

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

    # Initialize LLM adapter
    if config.llm_provider.lower() == "ollama":
        llm_adapter = OllamaAdapter(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
        )
    elif config.llm_provider.lower() == "openai":
        if not config.openai_api_key:
            logger.error("OPENAI_API_KEY not set")
            sys.exit(1)
        llm_adapter = OpenAIAdapter(
            api_key=config.openai_api_key,
            model=config.openai_model,
            timeout=config.request_timeout,
            max_retries=config.max_retries,
        )
    elif config.llm_provider.lower() == "mock":
        llm_adapter = MockLLMAdapter()
    else:
        logger.error(f"Unknown LLM provider: {config.llm_provider}")
        sys.exit(1)

    # Initialize retriever
    top_k = args.top_k if args.top_k is not None else config.top_k_results
    retriever = Retriever(
        vector_store=vector_store, embedding_adapter=embedding_adapter, top_k=top_k
    )

    # Check system health
    doc_count = vector_store.count()
    logger.info(f"System ready! Documents in store: {doc_count}")

    if doc_count == 0:
        logger.warning("No documents in vector store. Index some documents first!")
        sys.exit(1)

    # Query function
    def process_query(query_text: str):
        """Process a single query."""
        print(f"\n{'='*60}")
        print(f"Query: {query_text}")
        print(f"{'='*60}")

        # Retrieve documents
        results = retriever.retrieve(query=query_text, top_k=top_k)

        if not results:
            print("\nNo relevant documents found.")
            return

        # Format context
        context = retriever.format_context(results, include_metadata=False)

        # Build prompt
        prompt = RAG_WITH_SYSTEM.format(query=query_text, context=context)

        # Generate answer
        print("\nGenerating answer...")
        response = llm_adapter.generate(prompt)

        # Display answer
        print(f"\n{'-'*60}")
        print("Answer:")
        print(f"{'-'*60}")
        print(response.text)

        # Display sources if requested
        if args.show_sources:
            print(f"\n{'-'*60}")
            print(f"Sources ({len(results)} documents):")
            print(f"{'-'*60}")
            for i, result in enumerate(results, 1):
                print(f"\n[{i}] (score: {result.score:.3f})")
                print(f"  {result.content[:200]}...")
                if result.metadata:
                    print(f"  Metadata: {result.metadata}")

        # Display usage
        if response.usage:
            print(f"\n{'-'*60}")
            print(f"Token usage: {response.usage}")
            print(f"{'='*60}\n")

    # Interactive or single query mode
    if args.interactive or not args.query:
        print("\n" + "=" * 60)
        print("RAG Interactive Query Mode")
        print("=" * 60)
        print("Type your questions below. Type 'exit' or 'quit' to exit.\n")

        while True:
            try:
                query_text = input("Query: ").strip()
                if not query_text:
                    continue
                if query_text.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break

                process_query(query_text)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                continue
    else:
        # Single query mode
        process_query(args.query)


if __name__ == "__main__":
    main()

