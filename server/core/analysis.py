"""Core analysis engine for SurgicalAI.

Implements the main lesion analysis pipeline with structured outputs.
"""

import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timezone
import numpy as np
from PIL import Image

from .schemas import AnalysisRequest, AnalysisResponse, Diagnostics, DiagnosisProb
from .schemas import ABCDEAnalysis, FlapPlan, FlapParameters, FlapAlternative, Artifacts
from .llm import get_llm_analysis
from .utils import image_to_base64
import yaml


logger = logging.getLogger(__name__)

# Facial subunit mapping
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


async def analyze_lesion(request: AnalysisRequest, request_id: str) -> AnalysisResponse:
    """
    Main analysis pipeline for lesion images.
    
    Args:
        request: Analysis request with image and parameters
        request_id: Unique request identifier
        
    Returns:
        Structured analysis response
    """
    logger.info(f"Starting analysis for request {request_id}")
    
    # Create run directory
    runs_dir = Path(__file__).resolve().parents[2] / "runs" 
    run_dir = runs_dir / request_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Image preprocessing
        image = await preprocess_image(request.image_base64, run_dir)
        
        # 2. ROI extraction
        roi_image = extract_roi(image, request.roi)
        
        # 3. RSTL angle calculation (knowledge-based)
        rstl_angle = calculate_rstl_angle(request.site)
        
        # 4. Tension analysis
        tension_score = calculate_tension_score(roi_image, request.site)
        
        # 5. Diagnostic analysis
        offline_flag = getattr(request, 'offline', False)
        from .llm import get_offline_mode
        if get_offline_mode():
            offline_flag = True
        if offline_flag:
            diagnostics = await offline_diagnosis(roi_image, request.risk_factors)
        else:
            diagnostics = await llm_diagnosis(roi_image, request, request_id)
        
        # 6. ABCDE analysis
        abcde = calculate_abcde_score(roi_image)
        
        # 7. Flap planning
        flap_plan = plan_flap_reconstruction(
            roi_image, 
            request.site, 
            rstl_angle, 
            tension_score,
            diagnostics
        )
        
        # 8. Generate artifacts
        artifacts = await generate_artifacts(image, roi_image, request.roi, run_dir, rstl_angle, flap_plan.type if flap_plan else None, request.site)
        
        # 9. Gate based on knowledge and add citations
        knowledge = load_knowledge()
        citations = get_relevant_citations(diagnostics.top3[0].label if diagnostics.top3 else "general", knowledge)
        
        response = AnalysisResponse(
            diagnostics=diagnostics,
            abcde=abcde,
            rstl_angle_deg=rstl_angle,
            tension_score=tension_score,
            flap_plan=flap_plan,
            artifacts=artifacts,
            citations=citations,
            request_id=request_id
        )
        
        # Save analysis results
        await save_analysis_results(response, run_dir)
        
        logger.info(f"Analysis completed for request {request_id}")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed for request {request_id}: {e}", exc_info=True)
        raise


async def preprocess_image(image_data, run_dir: Path) -> Image.Image:
    """Preprocess and save original image."""
    if isinstance(image_data, str):
        # Decode base64
        import base64
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data
    
    from .utils import validate_image
    image, metadata = validate_image(image_bytes)
    
    # Save original
    original_path = run_dir / "original.jpg"
    image.save(original_path, "JPEG", quality=90)
    
    logger.info(f"Preprocessed image: {metadata}")
    return image


def extract_roi(image: Image.Image, roi_data) -> Image.Image:
    """Extract region of interest from image."""
    if isinstance(roi_data, dict):
        x = roi_data["x"]
        y = roi_data["y"] 
        width = roi_data["width"]
        height = roi_data["height"]
    else:
        x = roi_data.x
        y = roi_data.y
        width = roi_data.width
        height = roi_data.height
    
    # Convert normalized coordinates to pixels
    img_width, img_height = image.size
    left = int(x * img_width)
    top = int(y * img_height)
    right = int((x + width) * img_width)
    bottom = int((y + height) * img_height)
    
    roi_image = image.crop((left, top, right, bottom))
    logger.info(f"Extracted ROI: {roi_image.size}")
    return roi_image


def calculate_rstl_angle(site: Optional[str]) -> float:
    """Calculate RSTL angle for anatomical site using clinical knowledge."""
    knowledge = load_knowledge()
    table = knowledge.get('rstl_angles_deg', {})
    angle = float(table.get(site or '', 37.5))
    logger.info(f"RSTL angle for {site}: {angle}°")
    return angle


def calculate_tension_score(roi_image: Image.Image, site: Optional[str]) -> float:
    """Calculate tissue tension score based on image analysis."""
    # Simple heuristic based on site and image properties
    site_tension_factors = {
        "forehead": 0.3,
        "temple": 0.4,
        "eyelid_upper": 0.7,
        "eyelid_lower": 0.6,
        "nose_tip": 0.8,
        "nose_ala": 0.7,
        "cheek_medial": 0.4,
        "cheek_lateral": 0.3,
        "lip_upper": 0.8,
        "lip_lower": 0.7,
        "chin": 0.4,
    }
    
    base_tension = site_tension_factors.get(site, 0.5)
    
    # Adjust based on ROI size (larger defects = more tension)
    width, height = roi_image.size
    size_factor = min(1.0, (width * height) / (100 * 100))  # Normalize to 100x100 baseline
    
    tension_score = min(1.0, base_tension + (size_factor * 0.2))
    logger.info(f"Tension score for {site}: {tension_score:.3f}")
    return tension_score


async def offline_diagnosis(roi_image: Image.Image, risk_factors) -> Diagnostics:
    """Offline heuristic diagnosis (no LLM)."""
    # Simple heuristic scoring based on image properties
    image_array = np.array(roi_image)
    
    # Color variation analysis
    color_var = np.std(image_array) / 255.0
    
    # Size analysis
    width, height = roi_image.size
    size_score = min(1.0, max(width, height) / 100.0)
    
    # Heuristic probabilities
    if color_var > 0.3 and size_score > 0.5:
        # High variation + large size = concern
        probs = [
            DiagnosisProb(label="basal_cell_carcinoma", prob=0.45),
            DiagnosisProb(label="seborrheic_keratosis", prob=0.35),
            DiagnosisProb(label="nevus", prob=0.20),
        ]
    elif color_var > 0.2:
        # Moderate variation
        probs = [
            DiagnosisProb(label="seborrheic_keratosis", prob=0.50),
            DiagnosisProb(label="nevus", prob=0.30),
            DiagnosisProb(label="actinic_keratosis", prob=0.20),
        ]
    else:
        # Low variation - likely benign
        probs = [
            DiagnosisProb(label="nevus", prob=0.60),
            DiagnosisProb(label="seborrheic_keratosis", prob=0.25),
            DiagnosisProb(label="dermatofibroma", prob=0.15),
        ]
    
    return Diagnostics(
        top3=probs,
        notes="Heuristic analysis (offline mode). Consult dermatopathologist for definitive diagnosis."
    )


async def llm_diagnosis(roi_image: Image.Image, request: AnalysisRequest, request_id: str) -> Diagnostics:
    """LLM-powered diagnosis using structured outputs."""
    try:
        result = await get_llm_analysis(roi_image, request, request_id)
        return result
    except Exception as e:
        logger.warning(f"LLM analysis failed, falling back to offline: {e}")
        return await offline_diagnosis(roi_image, request.risk_factors)


def calculate_abcde_score(roi_image: Image.Image) -> ABCDEAnalysis:
    """Calculate ABCDE melanoma rule scores."""
    image_array = np.array(roi_image)
    
    # Simple heuristic implementation
    # In production, these would use proper image analysis algorithms
    
    # Asymmetry - compare halves
    height, width = image_array.shape[:2]
    left_half = image_array[:, :width//2]
    right_half = np.flip(image_array[:, width//2:], axis=1)
    
    if left_half.shape != right_half.shape:
        # Resize to match
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
    
    asymmetry = np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
    
    # Border irregularity - edge detection variance
    from scipy import ndimage
    edges = ndimage.sobel(np.mean(image_array, axis=2))
    border = np.std(edges) / 100.0  # Normalize
    
    # Color variation
    color = np.std(image_array) / 255.0
    
    # Diameter (normalized to 6mm baseline)
    diameter = min(1.0, max(roi_image.size) / 100.0)
    
    # Evolution (placeholder - would need temporal data)
    evolving = 0.0
    
    overall_score = (asymmetry + border + color + diameter + evolving) / 5.0
    
    return ABCDEAnalysis(
        asymmetry=min(1.0, asymmetry),
        border=min(1.0, border),
        color=min(1.0, color),
        diameter=diameter,
        evolving=evolving,
        overall_score=overall_score
    )


def plan_flap_reconstruction(
    roi_image: Image.Image,
    site: Optional[str], 
    rstl_angle: float,
    tension_score: float,
    diagnostics: Diagnostics
) -> FlapPlan:
    """Plan optimal flap reconstruction based on analysis and knowledge base."""
    knowledge = load_knowledge()
    width, height = roi_image.size
    defect_d_mm = max(width, height) * 0.1  # Rough mm conversion
    primary_dx = diagnostics.top3[0].label if diagnostics.top3 else "nevus"
    margin_mm = 2.0
    if "melanoma" in primary_dx:
        margin_mm = knowledge.get('margins_mm', {}).get('melanoma', {}).get('le_1mm', 10)
    elif "basal_cell" in primary_dx or "bcc" in primary_dx:
        margin_mm = knowledge.get('margins_mm', {}).get('bcc', {}).get('low_risk', 3)
    elif "squamous" in primary_dx or "scc" in primary_dx:
        margin_mm = knowledge.get('margins_mm', {}).get('scc', {}).get('low_risk', 4)

    # Flap selection logic (use requested flap if provided)
    flap_type = getattr(diagnostics, 'requested_flap', None)
    if not flap_type or flap_type == 'auto':
        # Auto selection by site/tension
        if site in ["eyelid_upper", "eyelid_lower", "lip_upper", "lip_lower", "nasal_ala"]:
            flap_type = "v-y"
        elif site in ["nose_tip", "nose_ala", "nose_dorsum"]:
            flap_type = "bilobed"
        elif tension_score > 0.7:
            flap_type = "rotation"
        elif site in ["cheek_lateral", "zygomatic"]:
            flap_type = "limberg"
        else:
            flap_type = "advancement"

    # Geometry and guideline info for each flap
    flap_info = {}
    if flap_type == "advancement":
        # Advancement Flap
        length = max(2*defect_d_mm, defect_d_mm+margin_mm)
        base_width = defect_d_mm+margin_mm
        ratio = min(length/base_width, 3.0)
        flap_info = {
            "geometry": f"Length ≥ 2×defect ({length:.1f}mm), Base width = defect+margin ({base_width:.1f}mm), Ratio ≤ 2:1 (safe), max 3:1",
            "indications": "Forehead, scalp, trunk, extremities with sufficient laxity. Closing defects adjacent to RSTL.",
            "contraindications": "Inelastic skin, over joints restricting ROM.",
            "howto": "Outline defect. Draw advancement flap along RSTL. Excise Burow's triangles. Dissect flap, advance, suture without torsion."
        }
        arc_deg = None
    elif flap_type == "rotation":
        # Rotation Flap
        angle = max(90, min(180, 90 + tension_score*90))
        radius = defect_d_mm/2.0
        arc_length = np.pi * radius * (angle/180)
        flap_length = max(4*defect_d_mm, arc_length)
        flap_info = {
            "geometry": f"Arc length = π×r×(θ/180) ({arc_length:.1f}mm), Flap length ≥ 4×defect ({flap_length:.1f}mm), Angle ≥90°, up to 180°.",
            "indications": "Large defects near convex surfaces (scalp, cheek, jawline). Tension redirection.",
            "contraindications": "Poor skin mobility, pivot near critical structures.",
            "howto": "Draw curvilinear incision, wide base. Undermine arc, advance flap, close secondary defect."
        }
        arc_deg = angle
        base_width = flap_length
    elif flap_type == "limberg":
        # Limberg Flap
        side = defect_d_mm
        flap_info = {
            "geometry": f"Diamond shape (60°/120°), Flap length = defect side ({side:.1f}mm)",
            "indications": "Cheek, zygomatic, areas with skin mobility.",
            "contraindications": "High tension risk, poor mobility.",
            "howto": "Draw rhombus, incise, transpose flap, close donor site."
        }
        arc_deg = 60
        base_width = side
    elif flap_type == "bilobed":
        # Bilobed Flap
        lobe1 = defect_d_mm
        lobe2 = 0.55*lobe1
        interlobe_angle = 47
        flap_info = {
            "geometry": f"First lobe = defect ({lobe1:.1f}mm), Second lobe = 0.55×first ({lobe2:.1f}mm), Interlobe angle ≈ 45–50°.",
            "indications": "Nasal tip, supratip.",
            "contraindications": "Inelastic areas, previous scarring.",
            "howto": "Draw lobes with ratios, incise, transpose, close donor site."
        }
        arc_deg = interlobe_angle
        base_width = lobe1
    elif flap_type == "v-y":
        # V-Y Advancement
        base_width = defect_d_mm
        length = 1.75*base_width
        flap_info = {
            "geometry": f"Base width = defect ({base_width:.1f}mm), Length = 1.75×base ({length:.1f}mm)",
            "indications": "Small defects in lip, eyelid, nasal ala, fingertip.",
            "contraindications": "Large defects needing excessive advancement.",
            "howto": "Mark V shape, incise, advance tip, suture as Y."
        }
        arc_deg = None
    elif flap_type == "keystone":
        # Keystone Flap (not in classic knowledge, but add geometry)
        base_width = defect_d_mm
        length = 2*base_width
        flap_info = {
            "geometry": f"Base width = defect ({base_width:.1f}mm), Length = 2×base ({length:.1f}mm)",
            "indications": "Defects on trunk/extremities with moderate laxity.",
            "contraindications": "Poor vascularity, irradiated tissue.",
            "howto": "Outline keystone shape, incise, advance, suture."
        }
        arc_deg = None
    elif flap_type == "nasolabial":
        base_width = defect_d_mm
        length = 2*base_width
        flap_info = {
            "geometry": f"Base width = defect ({base_width:.1f}mm), Length = 2×base ({length:.1f}mm)",
            "indications": "Nasal ala, upper/lower lip.",
            "contraindications": "Poor vascularity, previous scarring.",
            "howto": "Design flap on nasolabial fold, incise, advance, suture."
        }
        arc_deg = None
    elif flap_type == "paramedian_forehead":
        base_width = defect_d_mm
        length = 2*base_width
        flap_info = {
            "geometry": f"Base width = defect ({base_width:.1f}mm), Length = 2×base ({length:.1f}mm)",
            "indications": "Nasal tip, nasal ala.",
            "contraindications": "Supratrochlear injury risk.",
            "howto": "Design flap on forehead, incise, advance, suture."
        }
        arc_deg = None
    else:
        base_width = defect_d_mm
        arc_deg = None
        flap_info = {"geometry": "Not specified", "indications": "", "contraindications": "", "howto": ""}

    angle_to_rstl = abs(rstl_angle - 45.0)
    params = FlapParameters(
        defect_d_mm=defect_d_mm,
        margin_mm=margin_mm,
        arc_deg=arc_deg if arc_deg is not None else 0.0,
        base_width_mm=base_width,
        angle_to_rstl_deg=angle_to_rstl
    )

    # Alternative options
    alternatives = []
    for alt in ["rotation", "limberg", "v-y", "bilobed", "keystone", "advancement"]:
        if alt != flap_type:
            alternatives.append(FlapAlternative(type=alt, score=0.6))

    # Attach guideline info to params for frontend rendering
    params_dict = params.dict()
    params_dict.update(flap_info)

    return FlapPlan(
        type=flap_type,
        params=params,
        alternatives=alternatives
    )


async def generate_artifacts(
    original_image: Image.Image,
    roi_image: Image.Image, 
    roi_data,
    run_dir: Path,
    rstl_angle_deg: float = 0.0,
    flap_type: str = None,
    site: str = None
) -> Artifacts:
    """Generate visual artifacts (overlays, heatmaps)."""
    artifacts = Artifacts()
    try:
        # Generate ROI overlay with flap type and site
        overlay_image = generate_roi_overlay(original_image, roi_data, rstl_angle_deg, flap_type, site)
        # Export flap points for frontend
        overlay_flap_points = getattr(overlay_image, '_flap_points', None)
        if overlay_flap_points:
            artifacts.overlay_flap_points = overlay_flap_points
        overlay_path = run_dir / "overlay.png"
        overlay_image.save(overlay_path)
        artifacts.overlay_png_base64 = image_to_base64(overlay_image)
        
        # Generate heatmap centered on ROI centroid
        heatmap_image = generate_heatmap(roi_image)
        heatmap_path = run_dir / "heatmap.png"
        heatmap_image.save(heatmap_path)
        artifacts.heatmap_png_base64 = image_to_base64(heatmap_image)
        
        # Generate thumbnails
        thumbnail = roi_image.copy()
        thumbnail.thumbnail((128, 128))
        artifacts.thumbnails = [image_to_base64(thumbnail)]
    except Exception as e:
        logger.warning(f"Artifact generation failed: {e}")
    return artifacts


def generate_roi_overlay(image: Image.Image, roi_data, rstl_angle_deg: float = 0.0, flap_type: str = None, site: str = None) -> Image.Image:
    """Generate ROI overlay on original image, matching flap type and orientation."""
    from PIL import ImageDraw
    import math
    import numpy as np
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    
    if isinstance(roi_data, dict):
        x, y, width, height = roi_data["x"], roi_data["y"], roi_data["width"], roi_data["height"]
    else:
        x, y, width, height = roi_data.x, roi_data.y, roi_data.width, roi_data.height
    
    # Convert to pixel coordinates
    img_width, img_height = image.size
    left = int(x * img_width)
    top = int(y * img_height)
    right = int((x + width) * img_width)
    bottom = int((y + height) * img_height)
    
    # Draw ROI rectangle and centroid
    draw.rectangle([left, top, right, bottom], outline="red", width=3)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    r = max(2, int(min(image.size) * 0.01))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red")
    
    # Draw flap geometry overlay and collect points for frontend
    flap_points = []
    if flap_type == "rotation":
        if site in ["eyelid_upper", "eyelid_lower", "forehead"]:
            theta = 0.0
        elif site in ["jawline"]:
            theta = 45.0
        elif site in ["cheek_medial", "cheek_lateral"]:
            theta = 90.0
        else:
            theta = rstl_angle_deg
        angle = math.radians(theta)
        length = int(max(image.size) * 0.4)
        arc_radius = int(length * 0.8)
        arc_start = angle - math.radians(45)
        arc_end = angle + math.radians(90)
        for t in np.linspace(arc_start, arc_end, 30):
            x1 = int(cx + arc_radius * math.cos(t))
            y1 = int(cy + arc_radius * math.sin(t))
            flap_points.append([x1/img_width, y1/img_height])
            x2 = int(cx + arc_radius * math.cos(t + 0.1))
            y2 = int(cy + arc_radius * math.sin(t + 0.1))
            draw.line([x1, y1, x2, y2], fill="orange", width=3)
    elif flap_type == "v-y":
        if site in ["eyelid_upper", "eyelid_lower"]:
            theta = 0.0
        elif site in ["lip_upper", "lip_lower"]:
            theta = 90.0
        else:
            theta = rstl_angle_deg
        angle = math.radians(theta)
        length = int(max(image.size) * 0.3)
        dx = int(math.cos(angle) * length)
        dy = int(math.sin(angle) * length)
        # V points
        v1 = [cx + dx, cy + dy]
        v2 = [cx - dx, cy + dy]
        flap_points = [[cx/img_width, cy/img_height], [v1[0]/img_width, v1[1]/img_height], [cx/img_width, cy/img_height], [v2[0]/img_width, v2[1]/img_height]]
        draw.line([cx, cy, v1[0], v1[1]], fill="orange", width=3)
        draw.line([cx, cy, v2[0], v2[1]], fill="orange", width=3)
    # Add more flap types as needed
    # Draw RSTL dashed line
    try:
        angle = math.radians(float(rstl_angle_deg or 0.0))
        length = int(max(image.size) * 0.5)
        dx = int(math.cos(angle) * length)
        dy = int(math.sin(angle) * length)
        x1, y1 = cx - dx, cy - dy
        x2, y2 = cx + dx, cy + dy
        segments = 24
        for i in range(0, segments, 2):
            t1 = i / segments
            t2 = (i + 1) / segments
            sx1 = int(x1 + (x2 - x1) * t1)
            sy1 = int(y1 + (y2 - y1) * t1)
            sx2 = int(x1 + (x2 - x1) * t2)
            sy2 = int(y1 + (y2 - y1) * t2)
            draw.line([sx1, sy1, sx2, sy2], fill="blue", width=2)
    except Exception:
        pass
    # Return overlay and flap points for frontend
    overlay._flap_points = flap_points
    return overlay


def generate_heatmap(roi_image: Image.Image) -> Image.Image:
    """Generate attention heatmap centered on ROI centroid with Gaussian profile."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import io
    
    # Gaussian heatmap centered on ROI center
    width, height = roi_image.size
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width // 2, height // 2
    
    # Radial Gaussian based on distance from center
    sigma = max(width, height) / 4.0
    dist2 = ((x - center_x)**2 + (y - center_y)**2)
    heatmap = np.exp(-dist2 / (2 * sigma * sigma))
    
    # Apply colormap
    colored = cm.hot(heatmap)
    
    # Convert to PIL Image
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.imshow(colored)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def get_relevant_citations(diagnosis: str, knowledge: dict) -> List[str]:
    """Get citation strings based on diagnosis and knowledge base."""
    cites = knowledge.get('citations_strings', {})
    out: List[str] = []
    d = diagnosis.lower()
    if 'melanoma' in d:
        out.append(cites.get('melanoma_margins', 'NCCN Melanoma: margins per Breslow.'))
    if 'basal' in d or 'bcc' in d:
        out.append(cites.get('bcc_mohs', 'NCCN/AAD BCC: margins and Mohs indications.'))
    if 'squamous' in d or 'scc' in d:
        out.append(cites.get('scc_margins', 'BAD/NCCN SCC: margins by risk; Mohs for high risk.'))
    # Always include flap references
    out.append(cites.get('flap_refs', 'Facial flap parameters per standard texts.'))
    return out


_KNOWLEDGE_CACHE: Optional[dict] = None

def load_knowledge() -> dict:
    global _KNOWLEDGE_CACHE
    if _KNOWLEDGE_CACHE is not None:
        return _KNOWLEDGE_CACHE
    try:
        path = Path(__file__).with_name('clinical_knowledge.yaml')
        with open(path, 'r', encoding='utf-8') as f:
            _KNOWLEDGE_CACHE = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load clinical knowledge: {e}")
        _KNOWLEDGE_CACHE = {}
    return _KNOWLEDGE_CACHE


async def save_analysis_results(response: AnalysisResponse, run_dir: Path):
    """Save analysis results to disk."""
    # Save JSON response
    result_path = run_dir / "analysis.json"
    with open(result_path, 'w') as f:
        json.dump(response.dict(), f, indent=2, default=str)
    
    # Save metadata
    metadata = {
        "request_id": response.request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "primary_diagnosis": response.diagnostics.top3[0].label if response.diagnostics.top3 else None,
        "flap_type": response.flap_plan.type,
        "tension_score": response.tension_score
    }
    
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Analysis results saved to {run_dir}")
