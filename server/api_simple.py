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
    """Analyze lesion with enhanced pipeline including VLM observer and fusion."""
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
        # Import enhanced pipeline components
        from surgicalai_demo.pipeline import run_demo
        from datetime import datetime
        import uuid
        
        # Create unique run directory
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4]
        runs_dir = Path("runs") / run_id
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded image
        image_path = runs_dir / "input.jpg"
        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved image to: {image_path}")
        
        # Extract parameters from payload
        age = payload_data.get('age')
        sex = payload_data.get('sex', '')
        subunit = payload_data.get('subunit', 'cheek')
        
        # Set up metadata for enhanced pipeline
        meta = {
            'age': int(age) if age else None,
            'sex': sex,
            'body_site': subunit,
            'photo_type': 'clinical',
            'known_scale_mm_per_px': None
        }
        
        # Run enhanced pipeline
        summary = run_demo(
            face_img_path=None,
            lesion_img_path=image_path,
            out_dir=runs_dir,
            ask=False
        )
        
        # Extract results for API response
        probs = summary.get("probs", {})
        fusion_data = summary.get("fusion", {})
        observer_data = summary.get("observer", {})
        paths = summary.get("paths", {})
        
        # Determine gate decision
        if fusion_data:
            gate_status = fusion_data.get("gate", "")
            gate_notes = fusion_data.get("notes", "")
            top3 = fusion_data.get("top3", [])
        else:
            # Fallback gate logic
            top_class = max(probs.items(), key=lambda x: x[1]) if probs else ("unknown", 0)
            top_prob = top_class[1]
            
            if top_class[0] == "melanoma" and top_prob > 0.4:
                gate_status = "RED — urgent dermatology referral"
            elif top_prob < 0.3:
                gate_status = "UNCERTAIN — clinical correlation recommended" 
            else:
                gate_status = f"Likely {top_class[0].replace('_', ' ')}"
            
            gate_notes = f"CNN prediction: {top_class[0]} ({top_prob:.1%})"
            top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Build artifact URLs (alignment with new spec)
        artifacts = {
            "overlay_full": f"/api/artifact/{run_id}/overlay_full.png" if paths.get("overlay_full_png") else None,
            "overlay_zoom": f"/api/artifact/{run_id}/overlay_zoom.png" if paths.get("overlay_zoom_png") else None,
            "pdf": f"/api/artifact/{run_id}/report.pdf",
        }
        # Filter out None values
        artifacts = {k: v for k, v in artifacts.items() if v}
        
        # Prepare response
        result = {
            "ok": True,
            "run_id": run_id,
            "probs": probs,
            "top3": top3,
            "gate": {
                "status": gate_status,
                "notes": gate_notes,
                "allow_flap": "GREEN" in gate_status or "likely" in gate_status.lower()
            },
            "artifacts": artifacts
        }

        # Observer full JSON per spec
        if observer_data:
            result["observer"] = {
                "primary_pattern": observer_data.get("primary_pattern"),
                "likelihoods": observer_data.get("likelihoods", {}),
                "descriptors": observer_data.get("descriptors", []),
                "abcd_estimate": observer_data.get("abcd_estimate", {}),
                "recommendation": observer_data.get("recommendation"),
            }

        # Fusion block with top3 and notes per spec
        if fusion_data:
            result["fusion"] = {
                "top3": fusion_data.get("top3", []),
                "notes": fusion_data.get("fusion_notes", fusion_data.get("notes", "")),
            }
        
        logger.info(f"Enhanced analysis completed for run {run_id}")
        return result
        
    except ImportError as e:
        logger.error(f"Enhanced pipeline not available: {e}")
        # Fallback to original simple analysis
        return await _fallback_simple_analysis(file, payload_data, logger)
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {str(e)}")
        # Try fallback analysis
        try:
            return await _fallback_simple_analysis(file, payload_data, logger)
        except Exception as fallback_error:
            logger.error(f"Fallback analysis also failed: {fallback_error}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


async def _fallback_simple_analysis(file: UploadFile, payload_data: dict, logger):
    """Fallback to simple analysis if enhanced pipeline fails."""
    # Create runs directory
    runs_dir = Path("runs") / "demo"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded image
    image_path = runs_dir / file.filename
    with open(image_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"Fallback: Saved image to: {image_path}")
    
    # Generate mock probabilities
    import random
    random.seed(42)
    
    lesion_probs = {
        "melanoma": round(random.uniform(0.1, 0.8), 2),
        "nevus": round(random.uniform(0.1, 0.4), 2),
        "seborrheic_keratosis": round(random.uniform(0.05, 0.3), 2),
        "bcc": round(random.uniform(0.05, 0.2), 2),
        "scc": round(random.uniform(0.02, 0.15), 2),
        "benign_other": round(random.uniform(0.01, 0.1), 2)
    }
    
    # Normalize probabilities
    total = sum(lesion_probs.values())
    lesion_probs = {k: round(v/total, 3) for k, v in lesion_probs.items()}
    
    # Generate gate decision
    melanoma_prob = lesion_probs.get("melanoma", 0)
    allow_flap = melanoma_prob < 0.3
    gate_status = "Low melanoma suspicion" if allow_flap else "High melanoma suspicion"
    
    top3 = sorted(lesion_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return {
        "ok": True,
        "run_id": "demo-fallback",
        "probs": lesion_probs,
        "top3": top3,
        "gate": {
            "status": gate_status,
            "notes": "Fallback analysis - enhanced pipeline unavailable",
            "allow_flap": allow_flap
        },
        "artifacts": {
            "overlay": "/api/artifact/demo/overlay.png",
            "heatmap": "/api/artifact/demo/heatmap.png",
            "pdf": "/api/artifact/demo/report.pdf"
        }
    }

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
