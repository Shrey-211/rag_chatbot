# Multi-stage Dockerfile for RAG Chatbot API

# Base stage - common dependencies
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt


# Development/Local stage (default)
FROM base AS development

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY config.example.yaml ./config.example.yaml
COPY pyproject.toml .

# Create data directories
RUN mkdir -p /app/data/chroma /app/data/uploads

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with reload enabled for development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

