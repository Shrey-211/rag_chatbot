.PHONY: help install dev test lint format typecheck clean docker-build docker-build-fast docker-up docker-down index query

help:
	@echo "Available commands:"
	@echo "  make install           - Install production dependencies"
	@echo "  make dev               - Install development dependencies"
	@echo "  make test              - Run tests with coverage"
	@echo "  make lint              - Run linting (ruff)"
	@echo "  make format            - Format code with black"
	@echo "  make typecheck         - Run type checking with mypy"
	@echo "  make clean             - Clean build artifacts"
	@echo "  make docker-build      - Build Docker images (full rebuild)"
	@echo "  make docker-build-fast - Build with cache (much faster!)"
	@echo "  make docker-up         - Start Docker services"
	@echo "  make docker-down       - Stop Docker services"
	@echo "  make docker-restart    - Restart Docker services"
	@echo "  make index             - Index sample documents"
	@echo "  make query             - Run interactive query"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint:
	ruff check src/ api/ scripts/ tests/

format:
	black src/ api/ scripts/ tests/
	ruff check --fix src/ api/ scripts/ tests/

typecheck:
	mypy src/ api/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage build/ dist/ *.egg-info

docker-build:
	docker-compose build --no-cache

docker-build-fast:
	DOCKER_BUILDKIT=1 docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-restart:
	docker-compose down
	DOCKER_BUILDKIT=1 docker-compose up -d --build

index:
	python scripts/index_documents.py ./data/sample/

query:
	python scripts/query.py

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Development workflow
setup: dev
	cp .env.example .env
	cp config.example.yaml config.yaml
	mkdir -p data/sample data/chroma data/faiss data/uploads

all: format lint typecheck test
