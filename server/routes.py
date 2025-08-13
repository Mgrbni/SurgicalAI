"""API Routes for SurgicalAI

Implements the clean API contract:
- POST /api/analyze - single analysis endpoint
- POST /api/report - PDF generation endpoint
- GET /api/artifact/{path} - artifact serving
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .core.analysis import analyze_lesion
from .core.pdf import generate_report_pdf
from .core.schemas import AnalysisRequest, AnalysisResponse, ReportRequest
from .core.schemas import (
    TriageRequest, TriageResponse, DiagnosisProb, TriageDiagnostics,
    TriageOncology, TriageReconstruction, TriageSafety, TriageOverlay, TriageFollowUp
)
from .core.utils import generate_request_id, get_artifact_path, validate_image


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    payload: str = Form(...),
    offline: Optional[bool] = Form(False)
) -> AnalysisResponse:
    """
    Analyze a lesion image and return structured diagnosis and reconstruction plan.
    
    This is the primary analysis endpoint that replaces all scattered analyze endpoints.
    """
    request_id = generate_request_id()
    logger.info(f"Analysis request {request_id} started")
    
    # Add request ID to response headers
    headers = {"X-Request-ID": request_id}
    response.headers["X-Request-ID"] = request_id
    
    try:
        # Parse request payload
        import json
        try:
            payload_data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid JSON payload: {e}",
                headers=headers
            )
        
        # Validate required fields
        if "roi" not in payload_data:
            raise HTTPException(
                status_code=400,
                detail="Missing required field 'roi' in payload",
                headers=headers
            )
        
        # Validate image
        image_data = await file.read()
        image, metadata = validate_image(image_data, file.filename)
        
        # Basic validation
        roi = payload_data.get("roi")
        if not isinstance(roi, dict) or not all(k in roi for k in ("x","y","width","height")):
            raise HTTPException(status_code=400, detail="Invalid ROI; expected {x,y,width,height} normalized 0-1", headers=headers)
        for k in ("x","y","width","height"):
            try:
                roi[k] = float(roi[k])
            except Exception:
                raise HTTPException(status_code=400, detail=f"ROI.{k} must be a number", headers=headers)
        if roi["x"] < 0 or roi["y"] < 0 or roi["width"] <= 0 or roi["height"] <= 0:
            raise HTTPException(status_code=400, detail="ROI values must be positive and within 0-1", headers=headers)
        if roi["x"] + roi["width"] > 1.0 or roi["y"] + roi["height"] > 1.0:
            raise HTTPException(status_code=400, detail="ROI must lie within image bounds (normalized)", headers=headers)

        site = payload_data.get("site")
        from .core.analysis import FACIAL_SUBUNIT_LABELS
        if site and site not in FACIAL_SUBUNIT_LABELS:
            raise HTTPException(status_code=400, detail=f"Unknown facial_subunit '{site}'", headers=headers)

        # Create structured request
        analysis_request = AnalysisRequest(
            image_base64=image_data,
            roi=roi,
            site=site,
            risk_factors=payload_data.get("risk_factors", {}),
            offline=bool(offline or payload_data.get("offline") or False),
            provider=payload_data.get("provider"),
            model=payload_data.get("model")
        )
        
        # Perform analysis
        result = await analyze_lesion(analysis_request, request_id)
        
        logger.info(f"Analysis request {request_id} completed successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis request {request_id} failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
            headers=headers
        )


@router.post("/report")
async def generate_report_endpoint(
    request: Request,
    analysis_payload: str = Form(...),
    doctor_name: Optional[str] = Form(None)
) -> Response:
    """
    Generate and return a PDF report from analysis results.
    """
    request_id = generate_request_id()
    logger.info(f"Report generation request {request_id} started")
    
    try:
        # Parse JSON payload
        import json
        try:
            analysis_data = json.loads(analysis_payload) if isinstance(analysis_payload, str) else analysis_payload
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid analysis_payload JSON: {e}")

        # Create report request
        report_request = ReportRequest(
            analysis_payload=analysis_data,
            doctor_name=doctor_name or "Dr. Mehdi Ghorbani Karimabad"
        )
        
        # Generate PDF
        pdf_content = await generate_report_pdf(report_request, request_id)
        
        logger.info(f"Report generation request {request_id} completed")
        
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=surgical_report_{request_id}.pdf",
                "X-Request-ID": request_id
            }
        )
        
    except Exception as e:
        logger.error(f"Report generation request {request_id} failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}",
            headers={"X-Request-ID": request_id}
        )


@router.get("/artifact/{path:path}")
async def serve_artifact(path: str) -> FileResponse:
    """
    Serve artifact files from the runs directory.
    Implements secure path validation to prevent directory traversal.
    """
    try:
        artifact_path = get_artifact_path(path)
        
        if not artifact_path.exists() or not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        return FileResponse(
            path=artifact_path,
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "X-Content-Type-Options": "nosniff"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Artifact serving failed for path {path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Legacy compatibility endpoints (minimal implementation)
@router.get("/facial-subunits")
def get_facial_subunits():
    """Get available facial subunits for the UI."""
    from .core.analysis import FACIAL_SUBUNIT_LABELS
    return {"subunits": [{"value": k, "label": v} for k, v in FACIAL_SUBUNIT_LABELS.items()]}


@router.get("/metadata")
def get_metadata():
    """Get metadata for the UI (legacy compatibility)."""
    from .core.analysis import FACIAL_SUBUNIT_LABELS
    return {"facial_subunits": list(FACIAL_SUBUNIT_LABELS.keys())}


# === Minimal Clinical Demo: Triage endpoint ===

@router.post("/triage", response_model=TriageResponse)
async def triage_endpoint(payload: TriageRequest, request: Request, response: Response) -> TriageResponse:
    """Compute minimal clinical triage with vector overlay (no heatmap)."""
    request_id = generate_request_id()
    response.headers["X-Request-ID"] = request_id

    try:
        # Load clinical knowledge
        from .core.analysis import load_knowledge, calculate_rstl_angle
        knowledge = load_knowledge()

        # Decode image to get dimensions
        import base64
        from io import BytesIO
        from PIL import Image
        img_bytes = base64.b64decode(payload.image_base64.split(',')[-1]) if ',' in payload.image_base64 else base64.b64decode(payload.image_base64)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        w, h = img.size

        # Normalize/validate polygon
        poly = payload.roi_polygon
        if not isinstance(poly, list) or len(poly) < 3:
            raise HTTPException(status_code=400, detail="roi_polygon must be >= 3 points")
        # ensure normalized 0-1
        norm_poly = []
        for pt in poly:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                raise HTTPException(status_code=400, detail="roi_polygon points must be [x,y]")
            x, y = float(pt[0]), float(pt[1])
            if not (0 <= x <= 1 and 0 <= y <= 1):
                # if values look like pixels, normalize
                x = max(0, min(1, x / w))
                y = max(0, min(1, y / h))
            norm_poly.append([x, y])

        # Estimate defect size in mm via face-scale heuristic (interpupillary distance ~62mm)
        # Use polygon bounding box diagonal as diameter proxy
        xs = [p[0] * w for p in norm_poly]
        ys = [p[1] * h for p in norm_poly]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        bbox_w = maxx - minx
        bbox_h = maxy - miny
        import math
        defect_d_px = max(bbox_w, bbox_h)
        # assume face width ~ 2.5 * IPD; rough px/mm scale from image width
        ipd_mm = 62.0
        px_per_mm = (w / (2.5 * ipd_mm)) if w > 0 else 4.0
        defect_d_mm = max(2.0, defect_d_px / px_per_mm)

        # RSTL and safety
        rstl_angle = calculate_rstl_angle(payload.site)
        h_zone_sites = set((knowledge.get('h_zone', {}) or {}).get('subunits', []))
        in_h_zone = payload.site in h_zone_sites

        # Diagnostics (offline heuristic minimal)
        # Use size and risks to shape benign vs malignant prior
        benign_bias = 0.6 if defect_d_mm < 8 else 0.4
        risk_mod = 0.0
        if payload.risks.immunosuppressed: risk_mod += 0.1
        if payload.risks.ill_defined_borders: risk_mod += 0.1
        if payload.risks.recurrent_tumor: risk_mod += 0.1
        if in_h_zone: risk_mod += 0.05
        malignant_prior = min(0.6, 0.2 + risk_mod)
        bcc = 0.25 + (0.05 if in_h_zone else 0) + (0.05 if payload.risks.ill_defined_borders else 0)
        scc = 0.15 + (0.05 if payload.risks.immunosuppressed else 0)
        melanoma = 0.10 + (0.05 if payload.risks.recurrent_tumor else 0)
        nevus = benign_bias
        sk = 0.25
        # normalize
        raw = [('seborrheic_keratosis', sk), ('nevus', nevus), ('bcc', bcc), ('scc', scc), ('melanoma', melanoma)]
        total = sum(v for _, v in raw)
        probs = [(k, v/total) for k, v in raw]
        probs.sort(key=lambda x: x[1], reverse=True)
        top3 = [DiagnosisProb(label=k, prob=round(v, 4)) for k, v in probs[:3]]
        diagnostics = TriageDiagnostics(top3=top3, notes="Heuristic triage; confirm with histology.")

        # Oncologic margins (choose per top diagnosis)
        top_label = top3[0].label
        margins = knowledge.get('margins_mm', {})
        if top_label == 'melanoma':
            margin_mm = float(margins.get('melanoma', {}).get('le_1mm', 10))
        elif top_label == 'bcc':
            margin_mm = float(margins.get('bcc', {}).get('low_risk', 3 if not in_h_zone else 4))
        elif top_label == 'scc':
            margin_mm = float(margins.get('scc', {}).get('low_risk', 4))
        else:
            margin_mm = float(margins.get('benign', {}).get('default', 2))

        # Gates
        mohs_recommended = in_h_zone and (top_label in ['melanoma', 'bcc', 'scc'])
        defer_flap = mohs_recommended or (top_label == 'melanoma')
        oncology = TriageOncology(
            tumor_size_mm=round(defect_d_mm, 1),
            recommended_margin_mm=margin_mm,
            technique=("elliptical excision aligned with RSTL; consider primary closure"),
            gates={"mohs_recommended": bool(mohs_recommended), "defer_flap_until_histology": bool(defer_flap)}
        )

        # Reconstruction selection
        # Basic site-based choice; angle delta to RSTL
        angle_to_rstl = 12.0
        if payload.site in ['nose_tip','nose_ala']:
            primary = {"type":"bilobed","params":{"defect_mm":round(defect_d_mm,1)}}
            alts = [{"type":"nasolabial"},{"type":"paramedian_forehead"}]
        elif payload.site.startswith('eyelid') or payload.site in ['lip_upper','lip_lower']:
            primary = {"type":"v-y","params":{"advance_mm":round(defect_d_mm/2,1)}}
            alts = [{"type":"rotation"},{"type":"limberg"}]
        else:
            primary = {"type":"rotation","params":{"arc_deg":60,"base_width_mm":round(defect_d_mm*0.6,1)}}
            alts = [{"type":"v-y"},{"type":"limberg"}]
        reconstruction = TriageReconstruction(primary=primary, alternatives=alts, angle_to_rstl_deg=angle_to_rstl)

        # Safety alerts
        danger_map = {
            'upper_lip': ["Superior labial artery"],
            'lower_lip': ["Inferior labial artery"],
            'cheek_medial': ["Infraorbital nerve"],
            'cheek_lateral': ["Zygomatic/buccal branches"],
            'nose_tip': ["Supratrochlear vessels proximity"],
            'nose_ala': ["Angular artery"],
            'temple': ["Temporal branch facial nerve"],
        }
        risk_flags = [k for k, v in payload.risks.dict().items() if isinstance(v, bool) and v]
        safety = TriageSafety(h_zone=in_h_zone, nerve_vessel_alerts=danger_map.get(payload.site, []), risk_flags=risk_flags)

        # Vector overlay (ellipse + closure lines + flap anchor points)
        # Build an ellipse around bbox with margin
        margin_px = margin_mm * px_per_mm
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        major = max(bbox_w, bbox_h) + 2 * margin_px
        minor = max(major * 0.4, (max(bbox_w, bbox_h) * 0.5))
        # Sample ellipse points
        pts = []
        for t in [i * (2*math.pi/64) for i in range(64)]:
            ex = cx + (major/2.0) * math.cos(t)
            ey = cy + (minor/2.0) * math.sin(t)
            pts.append([max(0.0, min(1.0, ex / w)), max(0.0, min(1.0, ey / h))])

        # Closure line along RSTL axis through centroid
        angle = math.radians(rstl_angle)
        line_len = max(w, h) * 0.25
        dx = math.cos(angle) * line_len
        dy = math.sin(angle) * line_len
        line = [[max(0.0, min(1.0, (cx - dx)/w)), max(0.0, min(1.0, (cy - dy)/h))],
                [max(0.0, min(1.0, (cx + dx)/w)), max(0.0, min(1.0, (cy + dy)/h))]]

        # Flap geometry: simple rotation flap arc points with hinge at bbox corner
        hx, hy = minx, cy
        flap_pts = []
        for t in [i * (math.pi/12) for i in range(7)]:
            fx = hx + (defect_d_px*0.8) * math.cos(t)
            fy = hy - (defect_d_px*0.8) * math.sin(t)
            flap_pts.append([max(0.0, min(1.0, fx / w)), max(0.0, min(1.0, fy / h))])
        
        overlay = TriageOverlay(
            excision={"shape":"ellipse","points":pts},
            closure={"lines":[line]},
            flap={"type": primary["type"], "points": flap_pts}
        )

        # Follow-up
        follow = TriageFollowUp(
            pathology_review="confirm margins; if positive → re-excision/Mohs",
            wound_care="petrolatum + non-adherent dressing 48h; suture removal per site",
            surveillance="skin exam 6–12 months"
        )

        # Citations
        citations = [
            "NCCN BCC 2024", "AAD melanoma 2022", "Grabb & Smith 9e"
        ]

        # Render raster overlay for artifacts
        try:
            from PIL import ImageDraw
            overlay_img = img.copy()
            draw = ImageDraw.Draw(overlay_img)
            # ellipse polyline
            if pts:
                poly_px = [(p[0]*w, p[1]*h) for p in pts]
                draw.line(poly_px + [poly_px[0]], fill=(16,185,129), width=3)
            # closure line
            if line:
                draw.line([(line[0][0]*w,line[0][1]*h),(line[1][0]*w,line[1][1]*h)], fill=(37,99,235), width=3)
            # flap
            if flap_pts:
                flap_px = [(p[0]*w, p[1]*h) for p in flap_pts]
                draw.line(flap_px, fill=(245,158,11), width=3)
            import base64
            from io import BytesIO
            buf = BytesIO()
            overlay_img.save(buf, format='PNG')
            buf.seek(0)
            overlay_b64 = base64.b64encode(buf.read()).decode('ascii')
            artifacts = {"overlay_png_base64": overlay_b64, "thumbnails": []}
        except Exception:
            artifacts = None

        # Protocol/guideline summary
        proto_sources = [s.get('id') for s in (knowledge.get('sources') or []) if isinstance(s, dict) and s.get('id')]
        decisions = {
            "h_zone": bool(in_h_zone),
            "mohs_recommended": bool(mohs_recommended),
            "defer_flap_until_histology": bool(defer_flap),
            "rstl_angle_deg": float(rstl_angle),
            "recommended_margin_mm": float(margin_mm),
            "primary_flap": reconstruction.primary.get('type') if isinstance(reconstruction.primary, dict) else None
        }

        return TriageResponse(
            diagnostics=diagnostics,
            oncology=oncology,
            reconstruction=reconstruction,
            safety=safety,
            overlay=overlay,
            follow_up=follow,
            citations=citations,
            artifacts=artifacts,
            request_id=request_id,
            protocol={
                "name": "SurgicalAI Clinical Protocol",
                "version": str((knowledge.get('meta') or {}).get('version', '0.1.0')),
                "sources": proto_sources,
                "decisions": decisions
            },
            site=payload.site,
            rstl_angle_deg=rstl_angle
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Triage failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Triage failed")
