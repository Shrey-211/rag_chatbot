# Multi-stage Dockerfile for RAG Chatbot API - Optimized for build speed

# Base stage - common dependencies
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies with cache mount
# Includes Tesseract OCR and Poppler for image/PDF processing
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with cache mount
# Split into two steps: base dependencies first, then dev dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Development/Local stage (default)
FROM base AS development

# Copy application code (this layer changes most frequently)
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY config.example.yaml ./config.example.yaml
COPY pyproject.toml .

# Create data directories
RUN mkdir -p /app/data/chroma /app/data/uploads && \
    chmod -R 755 /app/data

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with reload enabled for development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Production stage
FROM base AS production

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY config.example.yaml ./config.example.yaml
COPY pyproject.toml .

# Create data directories with proper permissions
RUN mkdir -p /app/data/chroma /app/data/uploads && \
    chmod -R 755 /app/data

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run without reload for production
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
