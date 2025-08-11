"""HTTP API for SurgicalAI (clean local version).

This file was repaired: previous version was corrupted with concatenated
definitions causing syntax errors and preventing uvicorn from starting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SurgicalAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENT_DIR = Path(__file__).resolve().parents[1] / "client"
if CLIENT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")

FACIAL_SUBUNIT_LABELS = {
    "forehead": "Forehead",
    "temple": "Temple",
    "eyebrow": "Eyebrow",
    "eyelid_upper": "Upper Eyelid",
    "eyelid_lower": "Lower Eyelid",
    "canthus_medial": "Medial Canthus",
    "canthus_lateral": "Lateral Canthus",
    "nose_dorsum": "Nasal Dorsum",
    "nose_tip": "Nasal Tip",
    "nose_ala": "Nasal Ala",
    "cheek_medial": "Medial Cheek",
    "cheek_lateral": "Lateral Cheek",
    "zygomatic": "Zygomatic (Malar)",
    "lip_upper": "Upper Lip",
    "lip_lower": "Lower Lip",
    "philtrum": "Philtrum",
    "chin": "Chin",
    "jawline": "Jawline",
    "ear_helix": "Ear (Helix)",
    "ear_lobe": "Ear (Lobe)",
    "parotid_region": "Parotid Region",
    "neck_anterior": "Anterior Neck",
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/facial-subunits")
def get_facial_subunits():
    return {"subunits": [{"value": k, "label": v} for k, v in FACIAL_SUBUNIT_LABELS.items()]}


@app.get("/api/metadata")
def metadata():
    return {"facial_subunits": list(FACIAL_SUBUNIT_LABELS.keys())}


@app.api_route("/api/analyze", methods=["POST", "OPTIONS"], include_in_schema=True)
async def analyze_lesion(file: UploadFile | None = File(None), payload: str | None = Form(None)):
    # Fast path for preflight
    if file is None and payload is None:
        return {"ok": True}
    logger.info("Analyze request: %s (%s)", file.filename, file.content_type)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        payload_data = json.loads(payload)
        if not isinstance(payload_data, dict):
            raise ValueError("Payload must be object")
        if "subunit" not in payload_data:
            raise ValueError("Missing required field: subunit")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}") from e

    runs_dir = Path(__file__).resolve().parents[1] / "runs" / "demo"
    runs_dir.mkdir(parents=True, exist_ok=True)
    image_path = runs_dir / file.filename
    image_bytes = await file.read()
    image_path.write_bytes(image_bytes)

    import random
    random.seed(42)
    lesion_probs = {
        "melanoma": round(random.uniform(0.1, 0.8), 2),
        "nevus": round(random.uniform(0.1, 0.4), 2),
        "seb_keratosis": round(random.uniform(0.05, 0.3), 2),
        "bcc": round(random.uniform(0.05, 0.2), 2),
        "scc": round(random.uniform(0.02, 0.15), 2),
    }
    total = sum(lesion_probs.values())
    lesion_probs = {k: round(v / total, 2) for k, v in lesion_probs.items()}
    melanoma_prob = lesion_probs["melanoma"]
    allow_flap = melanoma_prob < 0.3
    gate_reason = "Low melanoma suspicion" if allow_flap else "High melanoma suspicion"
    gate_guidance = "Consider flap reconstruction" if allow_flap else "Biopsy/WLE first"

    # artifacts
    overlay_path = runs_dir / "overlay.png"
    heatmap_path = runs_dir / "heatmap.png"
    report_path = runs_dir / "report.pdf"
    for p, txt in [
        (overlay_path, b"PNG placeholder"),
        (heatmap_path, b"PNG placeholder"),
        (report_path, b"PDF placeholder"),
    ]:
        if not p.exists():
            p.write_bytes(txt)

    return {
        "ok": True,
        "lesion_probs": lesion_probs,
        "gate": {
            "allow_flap": allow_flap,
            "reason": gate_reason,
            "guidance": gate_guidance,
        },
        "artifacts": {
            "overlay_png": "/api/artifact/demo/overlay.png",
            "heatmap_png": "/api/artifact/demo/heatmap.png",
            "report_pdf": "/api/artifact/demo/report.pdf",
        },
    }


@app.get("/api/artifact/{path:path}")
async def get_artifact(path: str):
    artifact_path = Path(__file__).resolve().parents[1] / "runs" / path
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        artifact_path.resolve().relative_to((Path(__file__).resolve().parents[1] / "runs").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    return FileResponse(artifact_path)
