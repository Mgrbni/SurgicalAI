"""HTTP API for SurgicalAI - Simple working version."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SurgicalAI API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:5173", "http://127.0.0.1:8000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for client
app.mount("/", StaticFiles(directory="client", html=True), name="client")

# Curated facial subunit mapping
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
    """Return curated facial subunit options."""
    subunits = []
    for value, label in FACIAL_SUBUNIT_LABELS.items():
        subunits.append({"value": value, "label": label})
    return {"subunits": subunits}

@app.get("/api/metadata")
def metadata():
    """Legacy endpoint for backward compatibility."""
    return {"facial_subunits": list(FACIAL_SUBUNIT_LABELS.keys())}

@app.post("/api/analyze")
async def analyze_lesion(
    file: UploadFile = File(...),
    payload: str = Form(...)
):
    """Analyze lesion with multipart/form-data (file + JSON payload)."""
    logger.info(f"Received analyze request - file: {file.filename}, content_type: {file.content_type}")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        logger.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Parse JSON payload
        payload_data = json.loads(payload)
        logger.info(f"Parsed payload: {payload_data}")
        
        # Basic validation
        if not isinstance(payload_data, dict):
            raise HTTPException(status_code=400, detail="Payload must be a JSON object")
        if 'subunit' not in payload_data:
            raise HTTPException(status_code=400, detail="Missing required field: subunit")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in payload")
    except Exception as e:
        logger.error(f"Payload processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        # Create runs directory
        runs_dir = Path("runs") / "demo"
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded image
        image_path = runs_dir / file.filename
        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved image to: {image_path}")
        
        # Generate mock probabilities (in real implementation, this would call ML model)
        import random
        random.seed(42)  # For consistent demo results
        
        lesion_probs = {
            "melanoma": round(random.uniform(0.1, 0.8), 2),
            "nevus": round(random.uniform(0.1, 0.4), 2),
            "seb_keratosis": round(random.uniform(0.05, 0.3), 2),
            "bcc": round(random.uniform(0.05, 0.2), 2),
            "scc": round(random.uniform(0.02, 0.15), 2)
        }
        
        # Normalize probabilities
        total = sum(lesion_probs.values())
        lesion_probs = {k: round(v/total, 2) for k, v in lesion_probs.items()}
        
        # Generate gate decision based on melanoma probability
        melanoma_prob = lesion_probs.get("melanoma", 0)
        allow_flap = melanoma_prob < 0.3
        gate_reason = "Low melanoma suspicion" if allow_flap else "High melanoma suspicion"
        gate_guidance = "Consider flap reconstruction" if allow_flap else "Biopsy/WLE first"
        
        # Create mock artifacts
        overlay_path = runs_dir / "overlay.png"
        heatmap_path = runs_dir / "heatmap.png" 
        report_path = runs_dir / "report.pdf"
        
        # Create simple placeholder files
        if not overlay_path.exists():
            overlay_path.write_text("PNG placeholder")
        if not heatmap_path.exists():
            heatmap_path.write_text("PNG placeholder")
        if not report_path.exists():
            report_path.write_text("PDF placeholder")
        
        result = {
            "ok": True,
            "lesion_probs": lesion_probs,
            "gate": {
                "allow_flap": allow_flap,
                "reason": gate_reason,
                "guidance": gate_guidance
            },
            "artifacts": {
                "overlay_png": f"/api/artifact/demo/overlay.png",
                "heatmap_png": f"/api/artifact/demo/heatmap.png", 
                "report_pdf": f"/api/artifact/demo/report.pdf"
            }
        }
        
        logger.info(f"Analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/artifact/{path:path}")
async def get_artifact(path: str):
    """Serve artifact files from runs directory."""
    artifact_path = Path("runs") / path
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # Security check - ensure path is within runs directory
    try:
        artifact_path.resolve().relative_to(Path("runs").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(artifact_path)
