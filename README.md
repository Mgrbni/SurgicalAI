# SurgicalAI Demo (Grad‑CAM Alignment + Vision‑LLM Fusion)

> NOT FOR CLINICAL USE – Research / demonstration only.

This demo shows:
1. Exact inverse-mapped Grad‑CAM overlays (model space → original pixels) with ROI zoom.
2. Vision‑LLM Observer (OpenAI / Anthropic) returning structured JSON (no prose).
3. Neuro‑symbolic fusion boosting seborrheic keratosis (SK) when hallmark descriptors present.
4. PDF report with zoom + full overlays, observer & fusion summary tables.

## Quickstart

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .[dev]

# Run demo pipeline on sample image
python -m surgicalai_demo.pipeline --image data/samples/lesion.jpg --out runs/demo

# Or use Makefile shortcut (if make installed)
make demo

# Launch API server (simple)
python -m uvicorn server.api_simple:app --reload --port 8000
```

Open http://localhost:8000 and upload a lesion image. The client shows:
* ROI zoom overlay (switchable to full overlay)
* Observer (Vision‑LLM) descriptors & recommendation
* Fusion notes and top‑3 probabilities
* Downloadable PDF artifact

## Environment Variables

Set one provider (default openai):
```powershell
$env:VLM_PROVIDER = "openai"        # or "anthropic"
$env:OPENAI_API_KEY = "sk-..."      # if openai
$env:ANTHROPIC_API_KEY = "..."      # if anthropic
```

Optional:
```powershell
$env:OPENAI_MODEL = "gpt-4o-mini"
$env:ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
```

## Features Implemented

| Feature | Status |
|---------|--------|
| Grad‑CAM model→original exact warp | ✅ |
| ROI detection + zoom overlay | ✅ |
| Contours (0.3/0.5/0.7) + crosshair | ✅ |
| Observer JSON (primary_pattern, likelihoods, descriptors, ABCD, recommendation) | ✅ |
| Fusion: descriptor boosting & weighted blend | ✅ |
| PDF with observer + fusion tables | ✅ |
| API /api/analyze enriched response | ✅ |
| Unit tests: warp (<1px), overlay size, SK boost | ✅ |

## /api/analyze Response (excerpt)
```json
{
  "observer": {
    "primary_pattern": "seborrheic_keratosis",
    "likelihoods": {"seborrheic_keratosis": 0.42, ...},
    "descriptors": ["stuck-on appearance","waxy surface"],
    "abcd_estimate": {"asymmetry":"low","border":"sharp",...},
    "recommendation": "observe"
  },
  "fusion": {
    "top3": [["seborrheic_keratosis",0.31],["nevus",0.29],["scc",0.21]],
    "notes": "Boosted SK due to 'stuck-on','waxy' cues."
  },
  "artifacts": {
    "overlay_full": "/api/artifact/<run>/overlay_full.png",
    "overlay_zoom": "/api/artifact/<run>/overlay_zoom.png",
    "pdf": "/api/artifact/<run>/report.pdf"
  }
}
```

## Running Tests
```powershell
pytest -q
```

## Acceptance Checks
1. Cheek SK image: zoom overlay centered; contours align (no drift).  
2. Observer descriptors include ≥2 SK hallmarks → Fusion boosts SK probability.  
3. PDF shows zoom + full overlays, observer + fusion tables, disclaimer.  
4. Unit tests pass.

## Legacy Notes
## Fusion Logic Details

Fusion steps (`surgicalai_demo/fusion.py`):
1. Normalize CNN softmax outputs to probabilities.
2. Normalize VLM likelihoods → probabilities.
3. Descriptor rules:
  - If ≥2 SK descriptors among {stuck-on appearance, waxy surface, sharply demarcated, milia-like cysts, comedo-like openings} → +sk_descriptor_bonus to SK, small melanoma reduction (30% of bonus).
  - Melanoma descriptors (e.g., irregular pigment network, ulceration) → +melanoma_descriptor_bonus.
  - Similar optional boosts for BCC / SCC descriptor clusters.
4. Renormalize adjusted VLM probabilities.
5. Weighted blend: final = cnn_weight * cnn + vlm_weight * vlm_adjusted (weights auto-normalized).
6. Rank top‑k and apply gate logic (tie if top1-top2 ≤ tie_margin → UNCERTAIN).
7. Compile explanatory notes summarizing descriptor impacts + uncertainty.

Rationale: Keep CNN as primary signal (default 70%) while allowing structured visual cues to rectify systematic under-calling (SK) and elevate concerning melanoma patterns.
Older structured LLM endpoints retained (see `server/http_api.py`). New vision fusion path lives in `api_simple.py` (and mirrored in `api_advanced.py`).

---
## Dev Tasks
Format / lint:
```powershell
ruff check .
mypy surgicalai_demo
```
Type improvements and more robust model hooks are welcome.

---
## Checklist
- [ ] Set API key(s) & provider env vars
- [ ] Run pipeline / API
- [ ] Inspect overlays & PDF
- [ ] Review observer JSON & fusion notes

"Not for clinical use" is embedded in overlays & PDF.

# END README

## Installation

**Requires Python 3.8 or newer.**

Windows: [Download Python 3.10 or newer](https://www.python.org/downloads/windows/)

Linux/Mac: Install with pyenv (example for 3.11.9):
```bash
pyenv install 3.11.9
pyenv virtualenv 3.11.9 surgicalai-env
pyenv activate surgicalai-env
```

Note: 3D features have been disabled in this version (Open3D removed).

Endpoints
•GET /healthz — service health + current default model
•POST /api/infer — non-streaming; set "json_schema": true to enforce LesionReport schema
•POST /api/stream — streaming text for fast-feel UI

Example (Structured Outputs)

POST /api/infer
{
  "prompt": "Produce a LesionReport for a 12mm pigmented lesion over left malar area; include Grad-CAM hotspots and a rotation flap suggestion following Langer lines.",
  "json_schema": true
}

Returns a strict JSON object conforming to LesionReport.

Notes
•Uses responses.create/stream/parse (unified API surface).
•For browser realtime, mint ephemeral tokens server-side (not included here).
•Token/time caps are set in settings.yaml.

---

## AFTER WRITING FILES — VERIFY & RUN

1) Create virtualenv, install deps, run server:
  - `python -m venv .venv`
  - `. .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `pip install -e .`  # src-layout: enables imports like 'surgicalai.vision.models'
  - `python -m uvicorn server.http_api:app --reload --port 8000`  # use current venv

2) Open `http://localhost:8000`, type a prompt, click **Infer** or **Stream**.
3) Structured JSON requires the checkbox in the UI.

Print any errors you encounter and fix them automatically.

---

## GIT INIT & PUSH TO GITHUB

Now initialize a git repo inside `SurgicalAI/`, make the first commit, set the remote, and push:

- If I provided GitHub URL, user.name, and user.email, run these commands (shell):

```bash
cd SurgicalAI
git init
git branch -M main
git add .
git -c user.name="{{GIT_USER_NAME}}" -c user.email="{{GIT_USER_EMAIL}}" commit -m "feat: initial GPT-wired demo (FastAPI + Responses API + Structured Outputs + streaming)"
git remote add origin {{GITHUB_REPO_URL}}
git push -u origin main
```

Replace {{…}} placeholders with the values I gave you.
•If I didn’t provide values, print the exact commands with placeholders and instructions to paste them in my terminal.

Finally, echo a short checklist:
•cp .env.example .env and set OPENAI_API_KEY
•make demo
•Open http://localhost:8000
•Try a structured request (toggle “Structured JSON”)

End of instructions. Execute now.
