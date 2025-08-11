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

.PHONY: all install clean lint test server client api

all: install server client

server:
	$(PYTHON) -m uvicorn server.api_simple:app --port 8000 --reload

api:
	$(PYTHON) -m uvicorn server.api_advanced:app --reload --port 7860

client:
	$(PYTHON) -m http.server 5173 -d client

demo_cli: install
	@echo "==> Running legacy SurgicalAI CLI demo"
	$(PYTHON) -m surgicalai.cli demo --out runs/demo
	@echo "==> Done. Outputs in runs/demo/"

# New demo: start API + static client and perform health check
demo: install
	@echo "==> Launching SurgicalAI server (advanced) + client"
	-start "surgicalai_api" $(PYTHON) -m uvicorn server.api_simple:app --port 8000 --reload
	-start "surgicalai_client" $(PYTHON) -m http.server 5173 -d client
	@echo "Waiting for server to start..." && $(PYTHON) -c "import time; time.sleep(3)"
	@echo "==> Health check" && curl -fsS http://localhost:8000/api/health || (echo "Health check failed" && exit 1)
	@echo "Open http://localhost:5173 in your browser to use the demo"

# Enhanced demo: run multimodel consensus + PDF generation on sample image
consensus_demo: install
	@echo "==> Running dual-model consensus demo"
	$(PYTHON) - <<"PYEOF"
from pathlib import Path
import json
import cv2
from surgicalai_demo.multimodel import locate_lesion
from surgicalai_demo.guidelines import plan
from surgicalai_demo.risk_factors import adjust_probs
from surgicalai_demo.langer_lines import analyze_orientation_and_closure
import yaml

out = Path('runs/demo')
out.mkdir(parents=True, exist_ok=True)
img_path = Path('data') / 'sample.jpg'
if not img_path.exists():
    # fallback: pick any jpg
    for p in Path('data').rglob('*.jpg'):
        img_path = p; break
img = cv2.imread(str(img_path))
res = locate_lesion(img, out)
# Mock priors
priors = {"melanoma":0.2,"bcc":0.2,"scc":0.1,"nevus":0.3,"seborrheic_keratosis":0.15,"benign_other":0.05}
adj = adjust_probs(priors, {"uv_exposure":"high"})
rules = yaml.safe_load(open('data/oncology_rules.yaml','r',encoding='utf-8'))
center = tuple(res['consensus']['center'])
plan_dx = 'nevus_compound'
closure = analyze_orientation_and_closure('upper_lip', center, 6.0, rules)
plan_out = plan(plan_dx, 'upper_lip', 6.0, borders_clear=True, breslow_mm=None)
json.dump({"consensus":res, "adjusted":adj.posteriors, "closure":closure.__dict__, "plan":plan_out.__dict__}, open(out/'demo_summary.json','w'), indent=2)
print('Wrote runs/demo/demo_summary.json')
PYEOF
	@echo "==> Done. See runs/demo"

install:
	@echo "==> Creating venv and installing deps (runtime + dev)"
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt
	# Editable install with dev extras
	$(PIP) install -e .[dev]
	-$(PIP) install mediapipe || true

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
