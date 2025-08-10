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

.PHONY: demo install clean lint test

demo: install
	@echo "==> Running SurgicalAI demo"
	$(PYTHON) -m surgicalai.cli demo --cpu --offline-llm --out runs/demo
	@echo "==> Done. Outputs in runs/demo/"

install:
	@echo "==> Creating venv and installing deps"
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	# Minimal, stable deps for the demo
	$(PIP) install numpy opencv-python reportlab
	# Try to install mediapipe if platform supports; ignore failure (fallback to bbox midline)
	-$(PIP) install mediapipe

lint:
	@echo "==> Lint (ruff) and format check (black)"; \
	$(PIP) install ruff black >/dev/null; \
	$(VENV)/bin/ruff check surgicalai || true; \
	$(VENV)/bin/black --check surgicalai || true

test:
	@echo "==> No tests yet (prototype)"

clean:
	rm -rf $(VENV) build dist *.egg-info runs/demo
