"""
Quickstart Example: Complete RAG Pipeline

This example demonstrates:
1. Setting up adapters
2. Indexing documents
3. Querying the system
4. Getting answers with sources
"""

import os

from src.adapters.embedding.local import LocalTextEmbeddingAdapter
from src.adapters.llm.ollama import OllamaAdapter
from src.extractors.txt import TextExtractor
from src.retriever.retriever import Retriever
from src.utils.chunking import chunk_text
from src.utils.prompts import RAG_WITH_SYSTEM
from src.vectorstore.memory import InMemoryVectorStore


def main():
    """Run quickstart example."""
    print("=" * 60)
    print("RAG Chatbot Quickstart Example")
    print("=" * 60)

    # 1. Initialize components
    print("\n[1/5] Initializing components...")

    # Use local embeddings (no API key required)
    embedding_adapter = LocalTextEmbeddingAdapter(
        model_name="all-MiniLM-L6-v2", device="cpu"
    )

    # Use in-memory vector store (for demo)
    vector_store = InMemoryVectorStore()

    # Use Ollama LLM (make sure Ollama is running: `ollama serve`)
    llm_adapter = OllamaAdapter(base_url="http://localhost:11434", model="llama2")

    # Create retriever
    retriever = Retriever(
        vector_store=vector_store, embedding_adapter=embedding_adapter, top_k=3
    )

    print("✓ Components initialized")

    # 2. Index documents
    print("\n[2/5] Indexing sample document...")

    # Sample document
    document_text = """
    Retrieval-Augmented Generation (RAG) is a technique that combines 
    information retrieval with text generation. It works by first retrieving 
    relevant documents from a knowledge base, then using those documents as 
    context for generating accurate responses.
    
    RAG systems typically consist of three main components:
    1. A document store or vector database
    2. An embedding model for semantic search
    3. A large language model for generation
    
    The key advantage of RAG is that it allows language models to access 
    up-to-date information without requiring model retraining.
    """

    # Extract and chunk
    chunks = chunk_text(document_text, chunk_size=200, chunk_overlap=50)
    print(f"  Created {len(chunks)} chunks")

    # Embed chunks
    embeddings = embedding_adapter.embed_texts(chunks)
    print(f"  Generated embeddings: shape {embeddings.shape}")

    # Store in vector store
    chunk_ids = [f"doc1_chunk{i}" for i in range(len(chunks))]
    vector_store.upsert(chunk_ids, embeddings, chunks)
    print(f"✓ Indexed {len(chunks)} chunks")

    # 3. Query the system
    print("\n[3/5] Querying the system...")

    query = "What are the main components of a RAG system?"
    print(f"  Query: {query}")

    # Retrieve relevant documents
    results = retriever.retrieve(query, top_k=2)
    print(f"  Retrieved {len(results)} relevant chunks")

    # 4. Generate answer
    print("\n[4/5] Generating answer...")

    # Format context
    context = retriever.format_context(results)

    # Build prompt
    prompt = RAG_WITH_SYSTEM.format(query=query, context=context)

    # Generate response
    # Note: This requires Ollama to be running locally
    try:
        response = llm_adapter.generate(prompt)
        print("✓ Answer generated")
    except Exception as e:
        print(f"⚠ LLM error (is Ollama running?): {e}")
        print("  Using mock response for demo...")
        response_text = "Based on the context, a RAG system has three main components: a document store/vector database, an embedding model for semantic search, and a large language model for generation."
        from src.adapters.llm.base import LLMResponse

        response = LLMResponse(
            text=response_text, model="mock", usage={"total_tokens": 50}
        )

    # 5. Display results
    print("\n[5/5] Results:")
    print("-" * 60)
    print(f"Query: {query}")
    print("-" * 60)
    print(f"\nAnswer:\n{response.text}")
    print("\n" + "-" * 60)
    print("Sources:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] (score: {result.score:.3f})")
        print(f"  {result.content[:150]}...")

    print("\n" + "=" * 60)
    print("Quickstart complete!")
    print("=" * 60)

    # Next steps
    print("\nNext steps:")
    print("1. Try the CLI: python scripts/query.py --interactive")
    print("2. Start the API: uvicorn api.main:app --reload")
    print("3. Read docs: docs/getting_started.md")


if __name__ == "__main__":
    main()

