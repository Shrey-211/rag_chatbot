FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (cached layer)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements first for better caching
# This layer will only rebuild if requirements.txt changes
COPY requirements.txt .

# Install Python dependencies with pip cache
# Use buildkit cache mount to speed up rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code (changes frequently, so put it last)
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY config.example.yaml ./config.yaml

# Create data directories
RUN mkdir -p /app/data/chroma /app/data/uploads

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

