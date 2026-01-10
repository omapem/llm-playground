.PHONY: help install dev test lint format clean docs

help:
	@echo "LLM Playground Backend - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       Install dependencies"
	@echo "  make dev           Run development server"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linter (ruff)"
	@echo "  make format        Format code (black)"
	@echo "  make type-check    Run type checker (mypy)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Clean temporary files"
	@echo "  make docs          Generate documentation"

install:
	pip install -e ".[dev,training,inference,eval]"

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

lint:
	ruff check app/ tests/

format:
	black app/ tests/

type-check:
	mypy app/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist build *.egg-info

docs:
	@echo "Documentation generation to be implemented"
