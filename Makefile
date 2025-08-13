.PHONY: quickcheck demo fmt
quickcheck: ; python -m pip install -e ".[dev]" && pytest -q
demo: ; surgicalai demo assets/demo_face.jpg --out out
fmt: ; ruff check --fix . && black .
# SurgicalAI Makefile - Development and Production Tasks
PY := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python

# Detect Windows (Git Bash/PowerShell)
ifeq ($(OS),Windows_NT)
  PY := py -3
  VENV := .venv
  PIP := $(VENV)/Scripts/pip.exe
  PYTHON := $(VENV)/Scripts/python.exe
endif

.PHONY: help venv install dev demo test lint typecheck clean docker

# Default target
help:
	@echo "SurgicalAI Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make venv        - Create virtual environment"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Start development server"
	@echo ""
	@echo "Testing:"
	@echo "  make demo        - Run end-to-end demo"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run linting"
	@echo "  make typecheck   - Run type checking"
	@echo ""
	@echo "Production:"
	@echo "  make docker      - Build Docker image"
	@echo "  make clean       - Clean build artifacts"

# Virtual environment setup
venv:
	@echo "Creating virtual environment..."
	$(PY) -m venv $(VENV)
	@echo "Virtual environment created in $(VENV)/"
	@echo "Run 'make install' next"

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]
	@echo "Dependencies installed successfully"

# Development server (new clean API)
dev:
	@echo "Starting SurgicalAI development server..."
	@echo "Server will be available at: http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	$(PYTHON) -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Legacy server compatibility
server:
	$(PYTHON) -m uvicorn server.api_simple:app --port 8000 --reload

api:
	$(PYTHON) -m uvicorn server.api_advanced:app --reload --port 7860

# End-to-end demo
demo: install
	@echo "Running SurgicalAI demo..."
	$(PYTHON) demo_complete.py
	@echo "Demo completed. Check runs/ directory for results."

# Run tests
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v --cov=server --cov-report=html
	@echo "Test coverage report generated in htmlcov/"

# Linting
lint:
	@echo "Running linting..."
	$(PIP) install ruff >/dev/null 2>&1 || true
	$(PYTHON) -m ruff check server/ --fix
	$(PYTHON) -m ruff format server/

# Type checking
typecheck:
	@echo "Running type checks..."
	$(PIP) install mypy >/dev/null 2>&1 || true
	$(PYTHON) -m mypy server/ --ignore-missing-imports

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(VENV) build dist *.egg-info runs/ui __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	@echo "Clean completed"

# Docker build
docker:
	@echo "Building Docker image..."
	docker build -t surgicalai:latest -f Dockerfile .
	@echo "Docker image built: surgicalai:latest"
	@echo "Run with: docker run -p 8000:8000 surgicalai:latest"

# Production deployment preparation
prod-check: lint typecheck test
	@echo "Production readiness check..."
	@echo "✓ Linting passed"
	@echo "✓ Type checking passed"  
	@echo "✓ Tests passed"
	@echo "Ready for production deployment"

# Legacy demo compatibility
demo_cli: install
	@echo "==> Running legacy SurgicalAI CLI demo"
	$(PYTHON) -m surgicalai.cli demo --out runs/demo || echo "Legacy CLI not available"
	@echo "==> Done. Outputs in runs/demo/"

# Enhanced demo: run multimodel consensus + PDF generation on sample image
consensus_demo: install
	@echo "==> Running dual-model consensus demo"
	$(PYTHON) verify_advanced_system.py
	@echo "==> Done. See runs/demo"
