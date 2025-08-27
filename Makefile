# IUS Project Management
.PHONY: help install install-dev sync-requirements clean test lint

help:  ## Show this help message
	@echo "IUS Project Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt
	python -m spacy download en_core_web_lg

install-dev:  ## Install development dependencies  
	pip install -r requirements-dev.txt
	python -m spacy download en_core_web_lg

sync-requirements:  ## Sync requirements.txt with pyproject.toml
	python sync_requirements.py

clean:  ## Clean generated files
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:  ## Run tests
	python -m pytest

lint:  ## Run linting
	ruff check .
	ruff format --check .

fix-lint:  ## Fix linting issues
	ruff check --fix .
	ruff format .