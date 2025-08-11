# Makefile
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

.PHONY: all install clean lint test server client

all: install server client

server:
	$(PYTHON) -m uvicorn server.http_api:app --port 8000 --reload

client:
	$(PYTHON) -m http.server 5173 -d client

demo: install
	@echo "==> Running SurgicalAI demo (2D photo only)"
	$(PYTHON) -m surgicalai.cli demo --out runs/demo
	@echo "==> Done. Outputs in runs/demo/"

install:
	@echo "==> Creating venv and installing deps (2D photo only)"
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	# Install dependencies
	$(PIP) install numpy opencv-python reportlab
	$(PIP) install fastapi uvicorn[standard] python-multipart
	# Try to install mediapipe if platform supports; ignore failure (fallback to bbox midline)
	-$(PIP) install mediapipe
	# Install SurgicalAI in editable mode for src-layout
	$(PIP) install -e .

lint:
	@echo "==> Lint (ruff) and format check (black)"; \
	$(PIP) install ruff black >/dev/null; \
	$(VENV)/bin/ruff check surgicalai || true; \
	$(VENV)/bin/black --check surgicalai || true

test:
	@echo "==> Running tests"
	$(PYTHON) -m pytest tests/

clean:
	rm -rf $(VENV) build dist *.egg-info runs/ui
