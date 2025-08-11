# SurgicalAI — Photo-Only Surgical Planning

FastAPI server + tiny static client using OpenAI **Responses API** with **Structured Outputs** and streaming. Keys stay server‑side. Retries/backoff + token caps protect demo day.

## Quickstart (Photo-Only)

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -e .
pip install fastapi uvicorn[standard] python-multipart
python -m uvicorn server.http_api:app --reload --port 8000
```

Open http://localhost:8000 to use the client UI.

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
