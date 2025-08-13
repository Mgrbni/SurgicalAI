"""HTTP API for SurgicalAI (clean local version).

This file was repaired: previous version was corrupted with concatenated
definitions causing syntax errors and preventing uvicorn from starting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import io

try:
    from .settings import SETTINGS  # optional; silent if minimal deployment
except Exception:  # pragma: no cover
    class _Dummy:  # type: ignore
        photo_mode = True
        polycam_mode = False
    SETTINGS = _Dummy()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surgicalai")

app = FastAPI(title="SurgicalAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:5173",
    "http://localhost:8080",
        "http://127.0.0.1:8000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENT_DIR = Path(__file__).resolve().parents[1] / "client"

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


@app.get("/api/health")
def health_new():
    return {"ok": True}

@app.get("/health")
@app.get("/healthz")
def health_legacy():  # keep backward compatibility
    return {"status": "ok", "ok": True, "paths": ["/health", "/healthz", "/api/health"]}


@app.get("/api/facial-subunits")
def get_facial_subunits():
    return {"subunits": [{"value": k, "label": v} for k, v in FACIAL_SUBUNIT_LABELS.items()]}


@app.get("/api/metadata")
def metadata():
    return {"facial_subunits": list(FACIAL_SUBUNIT_LABELS.keys())}


def _generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{uuid.uuid4().hex[:4]}"

def _error(code: str, message: str, status: int = 400):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

@app.api_route("/api/analyze", methods=["POST", "OPTIONS"], include_in_schema=True)
async def analyze_lesion(
    request: Request,
    file: UploadFile | None = File(None),
    payload: str | None = Form(None),
    debug: int | None = None,
):
    # CORS preflight / form field introspection
    if request.method == "OPTIONS":
        return JSONResponse({"ok": True})
    if file is None:
        return _error("MISSING_IMAGE", "Upload an image.")
    if not file.content_type or not file.content_type.startswith("image/"):
        return _error("BAD_FILE_TYPE", "File must be an image")
    try:
        payload_data = json.loads(payload or '{}')
    except Exception as e:  # malformed JSON
        return _error("BAD_JSON", f"Invalid payload JSON: {e}")
    if not isinstance(payload_data, dict):
        return _error("BAD_JSON", "Payload root must be an object")
    if "subunit" not in payload_data:
        return _error("MISSING_FIELD", "Field 'subunit' required")

    run_id = _generate_run_id()
    runs_root = Path(__file__).resolve().parents[1] / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ext = ''.join(Path(file.filename).suffixes) or '.jpg'
    original_path = run_dir / f"original{ext}"
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        return _error("EMPTY_IMAGE", "Uploaded image is empty")
    if len(image_bytes) > 12 * 1024 * 1024:
        return _error("IMAGE_TOO_LARGE", "Image exceeds 12MB limit", status=413)
    # Validate open
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im.verify()
        # reopen for potential resize
        with Image.open(io.BytesIO(image_bytes)) as im2:
            if max(im2.size) > 5000:
                scale = 5000 / max(im2.size)
                new_size = (int(im2.size[0]*scale), int(im2.size[1]*scale))
                im2 = im2.resize(new_size)
                buf = io.BytesIO(); im2.save(buf, format='JPEG'); image_bytes = buf.getvalue()
    except UnidentifiedImageError:
        return _error("BAD_IMAGE", "Corrupted or unsupported image file")
    except Exception as e:
        logger.warning("Image validation error: %s", e)
        return _error("BAD_IMAGE", "Invalid image data")
    original_path.write_bytes(image_bytes)

    # Run inference pipeline
    from .pipeline import process_image
    proc = process_image(original_path, run_dir, debug=bool(debug))
    metrics = proc["metrics"]
    probs = metrics["probs"]
    top_label, _ = metrics["topk"][0]
    # Risk factor intake (future: pull from payload_data['risk_factors'])
    raw_risk = payload_data.get('risk_factors', {})
    raw_risk['site'] = payload_data.get('subunit')

    # Adjust probabilities via Bayesian-like scaling
    try:
        from .risk_factors import adjust_probabilities
        adjusted_probs = adjust_probabilities(probs, raw_risk)
    except Exception as e:
        logger.warning("Risk factor adjustment failed: %s", e)
        adjusted_probs = probs

    probs = {k: round(v,4) for k,v in adjusted_probs.items()}
    metrics['adjusted_probs'] = probs

    # Simplified diagnosis subset mapping (exclude rare / benign_other naming differences)
    diagnosis = {k: v for k, v in probs.items() if k in {"melanoma","bcc","scc","nevus"}}

    # Simple flap suggestion stub
    allow_flap = metrics['gate'].get('allow_flap', metrics['gate'].get('status') == 'OK')
    flap: Dict[str, Any]
    if allow_flap:
        flap = {
            "suggestion": "Rotation flap from lateral cheek" if 'cheek' in payload_data.get('subunit','') else "Advancement flap following relaxed skin tension lines",
            "tension_zone": "X-Y",
            "predicted_success": 0.82,
        }
    else:
        flap = {
            "suggestion": "Defer reconstruction – obtain histology first",
            "tension_zone": "n/a",
            "predicted_success": 0.0,
        }

    # Artifacts generation / normalization
    overlay_full_path = run_dir / 'overlay_full.png'
    overlay_zoom_path = run_dir / 'overlay_zoom.png'
    overlay_path = run_dir / 'overlay.png'  # Legacy, now points to zoom
    heatmap_top_path = run_dir / f'heatmap_{top_label}.png'
    heatmap_unified = run_dir / 'heatmap.png'
    
    # Create fallback overlays if enhanced version didn't generate them
    if not overlay_full_path.exists():
        overlay_full_path.write_bytes(b'PNG overlay_full placeholder')
    if not overlay_zoom_path.exists():
        overlay_zoom_path.write_bytes(b'PNG overlay_zoom placeholder')
    if not overlay_path.exists():
        overlay_path.write_bytes(b'PNG overlay placeholder')
    if not heatmap_top_path.exists():
        heatmap_top_path.write_bytes(b'PNG heatmap placeholder')
    if not heatmap_unified.exists():
        # duplicate top heatmap as generic heatmap.png
        heatmap_unified.write_bytes(heatmap_top_path.read_bytes())

    from .langer_lines import draw_langer_lines
    from .report_builder import build_enhanced_pdf
    try:
        # Placeholder lesion center (center of image). Future: consensus multi-model localization.
        with Image.open(original_path) as _im:
            cx, cy = _im.size[0]//2, _im.size[1]//2
        # Generate Langer lines overlay (saved as overlay_full if not already enhanced)
        langer_path = run_dir / 'overlay_full.png'
        try:
            draw_langer_lines(original_path, langer_path, lesion_center=(cx,cy))
        except Exception as le:
            logger.warning("Langer lines overlay failed: %s", le)
        # Enhanced PDF with guidelines + risk factors
        build_enhanced_pdf(
            run_id=run_id,
            run_dir=run_dir,
            probs=probs,
            gate=metrics['gate'],
            original_path=original_path,
            heatmap_path=heatmap_unified,
            overlay_path=overlay_path,
            risk_factors=raw_risk,
            lesion_center=(cx,cy),
            incision_angle=0.0,
        )
    except Exception as e:
        logger.warning("Enhanced report generation failed (%s): %s", run_id, e)

    # Plan JSON
    plan = {
        "run_id": run_id,
    "created_at": datetime.now(timezone.utc).isoformat(),
        "flap": flap,
        "gate": metrics['gate'],
        "subunit": payload_data.get('subunit'),
    }
    (run_dir / 'plan.json').write_text(json.dumps(plan, indent=2))

    # Manifest
    manifest = {
        "run_id": run_id,
        "diagnosis": diagnosis,
        "flap": flap,
        "artifacts": [
            {"name": "overlay_full", "path": f"runs/{run_id}/overlay_full.png"},
            {"name": "overlay_zoom", "path": f"runs/{run_id}/overlay_zoom.png"},
            {"name": "overlay", "path": f"runs/{run_id}/overlay.png"},  # Legacy
            {"name": "heatmap", "path": f"runs/{run_id}/heatmap.png"},
            {"name": "metrics", "path": f"runs/{run_id}/metrics.json"},
            {"name": "plan", "path": f"runs/{run_id}/plan.json"},
            {"name": "report", "path": f"runs/{run_id}/report.pdf"},
        ],
    }
    (run_dir / 'run.json').write_text(json.dumps(manifest, indent=2))

    # Backward-compatible legacy response keys (+ new contract fields)
    # Build heatmaps dict for legacy client
    legacy_heatmaps: Dict[str, str] = {}
    for lbl,_ in metrics['topk']:
        p = run_dir / f'heatmap_{lbl}.png'
        if p.exists():
            legacy_heatmaps[lbl] = f"/api/artifact/runs/{run_id}/{p.name}"

    response_payload: Dict[str, Any] = {
        "ok": True,
        "run_id": run_id,
        "diagnosis": diagnosis,
        "flap": flap,
    "lesion_probs": probs,  # legacy naming (now adjusted)
    "probs": probs,          # legacy alt naming
        "gate": metrics['gate'],
        "artifacts": {  # legacy structured map
            "overlay_full_png": f"/api/artifact/runs/{run_id}/overlay_full.png",
            "overlay_zoom_png": f"/api/artifact/runs/{run_id}/overlay_zoom.png",
            "overlay_png": f"/api/artifact/runs/{run_id}/overlay.png",  # Legacy
            "heatmap_png": f"/api/artifact/runs/{run_id}/heatmap.png",
            "report_pdf": f"/api/artifact/runs/{run_id}/report.pdf",
            "report_html": f"/api/artifact/runs/{run_id}/report.html",
            "metrics_json": f"/api/artifact/runs/{run_id}/metrics.json",
            "plan_json": f"/api/artifact/runs/{run_id}/plan.json",
            "heatmaps": legacy_heatmaps,
        },
        "artifacts_list": manifest['artifacts'],  # new contract
        "disclaimer": "Research prototype — NOT FOR CLINICAL USE",
    }
    if debug:
        response_payload['debug'] = metrics.get('debug')
    return JSONResponse(response_payload)


@app.get("/api/artifact/{path:path}")
async def get_artifact(path: str, request: Request):
    # Expect path like runs/<run_id>/file.ext
    root = Path(__file__).resolve().parents[1] / 'runs'
    if path.startswith('runs/'):
        rel = path[5:]
    else:
        rel = path
    artifact_path = root / rel
    # Security: ensure inside root
    try:
        artifact_path.resolve().relative_to(root.resolve())
    except Exception:
        logger.warning("Denied artifact traversal attempt: %s", path)
        raise HTTPException(status_code=403, detail="Access denied")
    if not artifact_path.exists() or not artifact_path.is_file():
        logger.info("Artifact 404: %s", artifact_path)
        raise HTTPException(status_code=404, detail="Artifact not found")
    logger.info("Artifact 200: %s", artifact_path)
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    return FileResponse(artifact_path, headers=headers)


@app.get("/debug/routes")
def list_routes():
    """List registered routes (debug helper to diagnose 404/405)."""
    out = []
    for r in app.router.routes:
        methods = sorted(getattr(r, "methods", []) or [])
        path = getattr(r, "path", None)
        if path:
            out.append({"path": path, "methods": methods})
    return {"routes": out}


@app.get("/api/last-usage")
async def get_last_usage(limit: int = 10):
    """Get recent usage logs from the runs directory."""
    try:
        runs_dir = Path(__file__).resolve().parents[1] / "runs"
        if not runs_dir.exists():
            return {"usage": []}
        
        # Get all run directories sorted by modification time (newest first)
        run_dirs = [
            d for d in runs_dir.iterdir() 
            if d.is_dir() and d.name.startswith("2025-")
        ]
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        usage_logs = []
        for run_dir in run_dirs[:limit]:
            try:
                # Try to read analysis results
                analysis_file = run_dir / "analysis.json"
                if analysis_file.exists():
                    with open(analysis_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract relevant info
                    log_entry = {
                        "id": run_dir.name,
                        "timestamp": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                        "subunit": data.get("payload", {}).get("subunit", "Unknown"),
                        "primary_diagnosis": None,
                        "success": data.get("ok", False)
                    }
                    
                    # Extract primary diagnosis from analysis
                    if data.get("analysis"):
                        if isinstance(data["analysis"], dict):
                            log_entry["primary_diagnosis"] = data["analysis"].get("primary_dx")
                        elif isinstance(data["analysis"], str):
                            # Sometimes analysis is a string
                            log_entry["analysis_text"] = data["analysis"][:100] + "..." if len(data["analysis"]) > 100 else data["analysis"]
                    
                    # Alternative: check diagnosis probabilities
                    if not log_entry["primary_diagnosis"] and data.get("diagnosis"):
                        # Find highest probability diagnosis
                        diagnoses = data["diagnosis"]
                        if isinstance(diagnoses, dict):
                            max_prob = 0
                            max_diag = None
                            for diag, prob in diagnoses.items():
                                if prob > max_prob:
                                    max_prob = prob
                                    max_diag = diag
                            log_entry["primary_diagnosis"] = max_diag
                            log_entry["confidence"] = max_prob
                    
                    usage_logs.append(log_entry)
                else:
                    # Basic entry without analysis
                    usage_logs.append({
                        "id": run_dir.name,
                        "timestamp": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                        "subunit": "Unknown",
                        "primary_diagnosis": "Analysis incomplete",
                        "success": False
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to parse usage log for {run_dir.name}: {e}")
                continue
        
        return {"usage": usage_logs}
        
    except Exception as e:
        logger.error(f"Failed to get usage logs: {e}")
        return {"usage": [], "error": str(e)}


# Mount static last so API routes take precedence
if CLIENT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")

# Global no-cache middleware
@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
