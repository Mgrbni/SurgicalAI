from __future__ import annotations

# features.py — lesion heuristics + overlay utils
# Developed by Dr. Mehdi Ghorbani (watermark added in save_overlay)

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import json
import numpy as np
import cv2
from PIL import Image
from skimage import filters, measure, morphology, exposure
from datetime import datetime


# ----------------------------- ABCDE model -----------------------------

@dataclass
class ABCDE:
    asymmetry: float
    border_irregularity: float
    color_variegation: float
    diameter_px: float
    elevation_satellite: float


def _read_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im)


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Very simple skin detection in YCrCb; Tier‑0 heuristic."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=256)
    mask = morphology.remove_small_objects(mask, min_size=256)
    return (mask.astype(np.uint8)) * 255


def _lesion_candidate(rgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Dark-object proposal inside skin using HSV-V Otsu + cleanup."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    v_eq = exposure.equalize_adapthist(v, clip_limit=0.01)
    v_u8 = (v_eq * 255).astype(np.uint8)
    v_skin = cv2.bitwise_and(v_u8, v_u8, mask=skin)
    thr = filters.threshold_otsu(v_skin[v_skin > 0]) if (v_skin > 0).any() else 128
    lesion = (v_skin < thr).astype(np.uint8) * 255  # darker than surround
    lesion = cv2.medianBlur(lesion, 5)
    lesion = morphology.remove_small_objects(lesion.astype(bool), 200).astype(np.uint8) * 255
    lesion = morphology.binary_closing(lesion, morphology.disk(3)).astype(np.uint8) * 255
    return lesion


def _largest_blob(mask: np.ndarray, min_frac: float = 0.001, max_frac: float = 0.30) -> np.ndarray:
    """Pick a sane lesion: largest component with area in [min,max] of image."""
    H, W = mask.shape[:2]
    labels = measure.label(mask > 0, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(mask)
    regions = list(measure.regionprops(labels))
    regions.sort(key=lambda r: r.area, reverse=True)
    area_min, area_max = min_frac * H * W, max_frac * H * W
    for r in regions:
        if area_min <= r.area <= area_max:
            return (labels == r.label).astype(np.uint8) * 255
    r = regions[0]  # fallback
    return (labels == r.label).astype(np.uint8) * 255


def _refine_with_grabcut(rgb: np.ndarray, blob: np.ndarray) -> np.ndarray:
    """Tighten the mask using GrabCut initialized from the blob’s bbox."""
    if blob.sum() == 0:
        return blob
    yx = np.column_stack(np.where(blob > 0))
    y0, x0 = yx.min(0)
    y1, x1 = yx.max(0)
    pad = 15
    H, W = blob.shape
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad); y1 = min(H - 1, y1 + pad)
    rect = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros(blob.shape, np.uint8) + cv2.GC_PR_BGD
    mask[blob > 0] = cv2.GC_PR_FGD

    try:
        cv2.grabCut(rgb, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return blob

    refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    refined = cv2.medianBlur(refined, 3)
    refined = morphology.remove_small_objects(refined.astype(bool), 200).astype(np.uint8) * 255
    return refined


def _asymmetry_score(blob: np.ndarray) -> float:
    ys, xs = np.where(blob > 0)
    if len(xs) < 20:
        return 0.0
    coords = np.column_stack([xs, ys]).astype(np.float32)
    mean = coords.mean(0)
    coords0 = coords - mean
    cov = np.cov(coords0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    rot = np.array([[major[0], -major[1]], [major[1], major[0]]])
    aligned = coords0 @ rot
    left = (aligned[:, 0] < 0).sum()
    right = (aligned[:, 0] >= 0).sum()
    area_diff = abs(left - right) / max(1, (left + right))
    return float(area_diff)


def _border_irregularity(blob: np.ndarray) -> float:
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if area <= 1:
        return 0.0
    return float((perim ** 2) / (4 * np.pi * area))


def _color_variegation(rgb: np.ndarray, blob: np.ndarray) -> float:
    """Counts hue clusters within mask (k=1..4) using k-means variance drop."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    Hvals = hsv[..., 0][blob > 0].reshape(-1, 1).astype(np.float32)
    if len(Hvals) < 50:
        return 0.0
    from sklearn.cluster import KMeans
    best_k, best_inertia = 1, None
    for k in range(1, 5):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(Hvals)
        if best_inertia is None or km.inertia_ < best_inertia * 0.7:
            best_k, best_inertia = k, km.inertia_
    return float(best_k - 1)  # 0..3 extra clusters


def _diameter_px(blob: np.ndarray) -> float:
    ys, xs = np.where(blob > 0)
    if len(xs) < 2:
        return 0.0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return float(max(x1 - x0, y1 - y0))


def _elevation_satellite(blob: np.ndarray) -> float:
    """Proxy: small secondary edge activity near lesion."""
    dil = cv2.dilate(blob, np.ones((7, 7), np.uint8), iterations=1)
    edge = cv2.Canny(dil, 50, 150)
    satellites = (edge > 0).sum() / 1000.0
    return float(min(satellites, 5.0))


def compute_abcde(rgb: np.ndarray) -> Dict[str, Any]:
    """Compute ABCDE features and lesion metrics from RGB image.
    
    Returns a dictionary instead of tuple for easier JSON serialization.
    """
    skin = _skin_mask(rgb)
    lesion0 = _lesion_candidate(rgb, skin)
    blob = _largest_blob(lesion0)
    blob = _refine_with_grabcut(rgb, blob)  # tighten mask

    A = _asymmetry_score(blob)
    B = _border_irregularity(blob)
    C = _color_variegation(rgb, blob)
    D = _diameter_px(blob)
    E = _elevation_satellite(blob)
    
    # Calculate center coordinates
    ys, xs = np.where(blob > 0)
    if len(xs) > 0:
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))
    else:
        center_x = float(rgb.shape[1] / 2)
        center_y = float(rgb.shape[0] / 2)
    
    return {
        "asymmetry": float(A),
        "border_irregularity": float(B),
        "color_variegation": float(C),
        "diameter_px": float(D),
        "elevation_satellite": float(E),
        "center_xy": [center_x, center_y],
        "mask": blob,
        "area_px": float(np.sum(blob > 0))
    }


# --------------------------- Attention/overlay ---------------------------

def _attention_map(rgb: np.ndarray, blob: np.ndarray) -> np.ndarray:
    """Attention = inside-distance + local darkness contrast; normalized [0,1]."""
    H, W = blob.shape
    inside = cv2.distanceTransform((blob > 0).astype(np.uint8), cv2.DIST_L2, 5)
    inside = inside / (inside.max() + 1e-6) if inside.max() > 0 else np.zeros_like(inside, dtype=np.float32)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    L_blur = cv2.GaussianBlur(L, (0, 0), 3)
    contrast = L_blur - L  # darker → positive
    band = cv2.dilate((blob > 0).astype(np.uint8), np.ones((21, 21), np.uint8), 1)
    c_vals = contrast[band > 0]
    if c_vals.size > 0:
        lo, hi = np.percentile(c_vals, [5, 95])
        contrast = np.clip((contrast - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        contrast = np.zeros_like(inside, dtype=np.float32)

    att = 0.6 * inside + 0.4 * contrast
    att = att * (band > 0)
    att = cv2.GaussianBlur(att, (0, 0), 2)
    if att.max() > 0:
        att = att / att.max()
    return att.astype(np.float32)


def save_overlay(rgb: np.ndarray, heatmap: Optional[np.ndarray], out_path: Path, plan_data: Optional[Dict] = None):
    """Save lesion outline + attention heat overlay with surgical planning overlays.
    
    This is a compatibility function. For advanced heatmap generation, use gradcam.py
    """
    if heatmap is None:
        # Use ABCDE-based attention map as fallback
        abcde_data = compute_abcde(rgb)
        blob = abcde_data.get("mask", np.zeros(rgb.shape[:2], dtype=np.uint8))
        heatmap = _attention_map(rgb, blob)
    
    vis = rgb.copy()

    # Draw lesion contour if we have a binary mask
    if heatmap is not None and len(np.unique(heatmap)) == 2:
        # Binary mask case
        blob = (heatmap > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lesion_center = None
        if cnts:
            cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)
            
            # Calculate lesion center for surgical overlays
            largest_contour = max(cnts, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                lesion_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        # Use attention map for blending
        att = _attention_map(rgb, blob)
    else:
        # Continuous heatmap case
        att = heatmap.astype(np.float32)
        if att.max() > 0:
            att = att / att.max()
        
        # Find center from peak activation
        peak_y, peak_x = np.unravel_index(np.argmax(att), att.shape)
        lesion_center = (int(peak_x), int(peak_y))

    # Apply heatmap overlay
    heat = cv2.applyColorMap((att * 255).astype(np.uint8), cv2.COLORMAP_JET)
    band3 = np.dstack([att > 0] * 3).astype(np.uint8)
    blended = np.where(band3 > 0, (0.65 * vis + 0.35 * heat).astype(np.uint8), vis)

    # Feather edges
    edge_mask = cv2.dilate((att > 0.1).astype(np.uint8), np.ones((3, 3), np.uint8), 1)
    edge = edge_mask - (att > 0.1).astype(np.uint8)
    blended[edge > 0] = (0.5 * blended[edge > 0] + 0.5 * np.array([255, 255, 255])).astype(np.uint8)

    # Add surgical planning overlays if plan data provided
    if plan_data and lesion_center:
        blended = _add_surgical_overlays(blended, lesion_center, plan_data)

    # Footer watermark
    h, w = blended.shape[:2]
    footer = "Developed by Dr. Mehdi Ghorbani"
    cv2.putText(blended, footer, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    Image.fromarray(blended).save(out_path)


def _add_surgical_overlays(img: np.ndarray, lesion_center: Tuple[int, int], plan_data: Dict) -> np.ndarray:
    """Add RSTL lines and flap planning overlays to the image."""
    overlay = img.copy()
    h, w = overlay.shape[:2]
    cx, cy = lesion_center
    
    # Extract planning data
    detailed_plan = plan_data.get('detailed', {})
    if not detailed_plan or detailed_plan.get('error'):
        return overlay
    
    rstl_angle = detailed_plan.get('incision_orientation_deg')
    candidates = detailed_plan.get('candidates', [])
    
    # 1. Draw RSTL orientation line (thin gold line)
    if rstl_angle is not None:
        # Convert angle to radians (0° = horizontal, 90° = vertical)
        angle_rad = np.radians(rstl_angle)
        
        # Draw line through lesion center along RSTL
        line_length = min(w, h) // 4
        dx = int(line_length * np.cos(angle_rad))
        dy = int(line_length * np.sin(angle_rad))
        
        # RSTL line in gold
        cv2.line(overlay, 
                (cx - dx, cy - dy), 
                (cx + dx, cy + dy), 
                (0, 215, 255),  # Gold color (BGR)
                2)
        
        # Add RSTL label
        label_x = cx + dx + 10
        label_y = cy + dy
        cv2.putText(overlay, f"RSTL {rstl_angle}°", 
                   (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 215, 255), 1, cv2.LINE_AA)
    
    # 2. Draw flap design hints for top candidate
    if candidates:
        top_candidate = candidates[0]
        flap_type = top_candidate.get('flap', '')
        
        # Different overlay patterns for different flap types
        if 'bilobed' in flap_type.lower():
            _draw_bilobed_overlay(overlay, lesion_center, top_candidate)
        elif 'rotation' in flap_type.lower() or 'mustarde' in flap_type.lower():
            _draw_rotation_overlay(overlay, lesion_center, top_candidate)
        elif 'advancement' in flap_type.lower():
            _draw_advancement_overlay(overlay, lesion_center, top_candidate, rstl_angle)
        elif 'forehead' in flap_type.lower():
            _draw_forehead_flap_overlay(overlay, lesion_center, top_candidate)
    
    # 3. Add legend for surgical overlays
    _add_surgical_legend(overlay)
    
    return overlay


def _draw_bilobed_overlay(img: np.ndarray, center: Tuple[int, int], candidate: Dict):
    """Draw bilobed flap design overlay."""
    cx, cy = center
    
    # Get bilobed parameters from candidate data
    vector_data = candidate.get('vector', {})
    total_arc = vector_data.get('total_arc_max', 100)
    lobe1_angle = vector_data.get('lobe1_deg', 45)
    lobe2_angle = vector_data.get('lobe2_deg', 45)
    
    # Estimate defect size and flap proportions
    radius = 30  # Base radius for visualization
    
    # Draw defect circle
    cv2.circle(img, center, radius, (0, 0, 255), 2)  # Red for defect
    
    # Draw first lobe (60% of defect size)
    lobe1_center = (cx + int(radius * 1.5 * np.cos(np.radians(lobe1_angle))),
                   cy + int(radius * 1.5 * np.sin(np.radians(lobe1_angle))))
    cv2.circle(img, lobe1_center, int(radius * 0.6), (255, 165, 0), 2)  # Orange
    
    # Draw second lobe (50% of first lobe)
    total_angle = lobe1_angle + lobe2_angle
    lobe2_center = (cx + int(radius * 2.2 * np.cos(np.radians(total_angle))),
                   cy + int(radius * 2.2 * np.sin(np.radians(total_angle))))
    cv2.circle(img, lobe2_center, int(radius * 0.3), (255, 165, 0), 2)  # Orange
    
    # Draw connecting arcs
    cv2.line(img, center, lobe1_center, (255, 165, 0), 1)
    cv2.line(img, lobe1_center, lobe2_center, (255, 165, 0), 1)
    
    # Add label
    cv2.putText(img, "Bilobed", (cx + 40, cy - 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1, cv2.LINE_AA)


def _draw_rotation_overlay(img: np.ndarray, center: Tuple[int, int], candidate: Dict):
    """Draw rotation flap design overlay."""
    cx, cy = center
    radius = 40
    
    # Draw defect
    cv2.circle(img, center, radius, (0, 0, 255), 2)
    
    # Draw rotation arc (lateral to medial)
    arc_start = (cx + radius * 2, cy - radius)
    arc_end = (cx - radius, cy + radius)
    
    # Draw curved line to represent rotation
    points = []
    for t in np.linspace(0, 1, 20):
        # Bezier-like curve
        x = int(arc_start[0] * (1-t) + arc_end[0] * t + radius * np.sin(t * np.pi))
        y = int(arc_start[1] * (1-t) + arc_end[1] * t)
        points.append((x, y))
    
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (0, 255, 0), 2)  # Green
    
    # Arrow to show direction
    cv2.arrowedLine(img, (cx + radius * 2, cy), (cx + radius, cy), (0, 255, 0), 2)
    
    cv2.putText(img, "Rotation", (cx + 50, cy), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


def _draw_advancement_overlay(img: np.ndarray, center: Tuple[int, int], candidate: Dict, rstl_angle: Optional[float]):
    """Draw advancement flap overlay."""
    cx, cy = center
    radius = 35
    
    # Draw defect
    cv2.circle(img, center, radius, (0, 0, 255), 2)
    
    # Draw advancement direction (along RSTL if available)
    if rstl_angle is not None:
        angle_rad = np.radians(rstl_angle)
        advance_start = (cx - int(radius * 2 * np.cos(angle_rad)),
                        cy - int(radius * 2 * np.sin(angle_rad)))
        
        # Draw advancement arrow
        cv2.arrowedLine(img, advance_start, center, (0, 255, 255), 2)  # Cyan
        
        # Draw Burow triangles
        triangle_offset = radius * 1.5
        triangle1 = (cx - int(triangle_offset * np.cos(angle_rad + np.pi/2)),
                    cy - int(triangle_offset * np.sin(angle_rad + np.pi/2)))
        triangle2 = (cx - int(triangle_offset * np.cos(angle_rad - np.pi/2)),
                    cy - int(triangle_offset * np.sin(angle_rad - np.pi/2)))
        
        # Small triangles for Burow excisions
        for triangle in [triangle1, triangle2]:
            pts = np.array([triangle, 
                           (triangle[0] + 10, triangle[1] + 10),
                           (triangle[0] - 10, triangle[1] + 10)], np.int32)
            cv2.drawContours(img, [pts], 0, (0, 255, 255), 1)
    
    cv2.putText(img, "Advancement", (cx + 40, cy + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)


def _draw_forehead_flap_overlay(img: np.ndarray, center: Tuple[int, int], candidate: Dict):
    """Draw paramedian forehead flap overlay."""
    cx, cy = center
    
    # Draw nasal defect
    cv2.circle(img, center, 25, (0, 0, 255), 2)
    
    # Draw pedicle line toward forehead (simplified)
    pedicle_end = (cx - 20, cy - 80)  # Approximate supratrochlear location
    cv2.line(img, center, pedicle_end, (255, 0, 255), 3)  # Magenta
    
    # Draw forehead donor area (rectangular approximation)
    donor_rect = (pedicle_end[0] - 30, pedicle_end[1] - 40, 60, 80)
    cv2.rectangle(img, 
                 (donor_rect[0], donor_rect[1]), 
                 (donor_rect[0] + donor_rect[2], donor_rect[1] + donor_rect[3]), 
                 (255, 0, 255), 2)
    
    cv2.putText(img, "Forehead Flap", (cx + 30, cy - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "2-Stage", (cx + 30, cy - 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1, cv2.LINE_AA)


def _add_surgical_legend(img: np.ndarray):
    """Add legend for surgical overlay colors."""
    h, w = img.shape[:2]
    
    # Legend background
    legend_w, legend_h = 200, 100
    legend_x, legend_y = w - legend_w - 10, 10
    
    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + legend_w, legend_y + legend_h), 
                 (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Legend text
    legend_items = [
        ("RSTL Lines", (0, 215, 255)),      # Gold
        ("Defect", (0, 0, 255)),            # Red  
        ("Flap Design", (255, 165, 0)),     # Orange
        ("Rotation", (0, 255, 0)),          # Green
        ("Advancement", (0, 255, 255))      # Cyan
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y + 15 + i * 15
        cv2.circle(img, (legend_x + 10, y_pos), 3, color, -1)
        cv2.putText(img, label, (legend_x + 20, y_pos + 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def dump_metrics(
    out_json: Path,
    img_path: str,
    mm_per_px: Optional[float],
    metrics_px: Dict[str, Any],
    metrics_mm: Optional[Dict[str, Any]],
    abcde: Dict[str, Any]
):
    """Save comprehensive metrics to JSON file."""
    payload = {
        "source_image": img_path,
        "scale_mm_per_px": mm_per_px,
        "metrics_px": metrics_px,
        "metrics_mm": metrics_mm,
        "abcde_features": abcde,
        "timestamp": str(datetime.now().isoformat())
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))


# ------------------------ Scale estimation helpers -----------------------

@dataclass
class ScaleEstimate:
    mm_per_px: Optional[float]
    source: str            # "ipd", "wtw", "manual", "unknown"
    details: Dict[str, float]


IPD_MM_DEFAULT = 63.0       # adult interpupillary distance (mean)
WTW_MM_DEFAULT = 11.7       # white-to-white corneal diameter (mean)


def estimate_scale(rgb: np.ndarray) -> Optional[float]:
    """Estimate mm per pixel scale from face image (placeholder implementation)."""
    # This is a simplified version that returns a reasonable default
    # In practice, this would use face detection/landmark detection
    # For now, assume typical smartphone photo scale
    h, w = rgb.shape[:2]
    
    # Rough heuristic: assume image width represents ~10-15cm face width
    estimated_face_width_mm = 120  # mm
    estimated_mm_per_px = estimated_face_width_mm / w
    
    # Clamp to reasonable bounds
    if 0.01 <= estimated_mm_per_px <= 0.5:
        return estimated_mm_per_px
    return None
def estimate_scale(rgb: np.ndarray) -> Optional[float]:
    """Estimate mm per pixel scale from face image (placeholder implementation)."""
    # This is a simplified version that returns a reasonable default
    # In practice, this would use face detection/landmark detection
    # For now, assume typical smartphone photo scale
    h, w = rgb.shape[:2]
    
    # Rough heuristic: assume image width represents ~10-15cm face width
    estimated_face_width_mm = 120  # mm
    estimated_mm_per_px = estimated_face_width_mm / w
    
    # Clamp to reasonable bounds
    if 0.01 <= estimated_mm_per_px <= 0.5:
        return estimated_mm_per_px
    return None


def defect_metrics_px_to_mm(
    metrics_px: Dict[str, Any],
    mm_per_px: Optional[float]
) -> Optional[Dict[str, Any]]:
    """Convert pixel metrics to mm using the scale factor."""
    if not mm_per_px or not metrics_px:
        return None
        
    metrics_mm = {}
    
    # Convert diameter
    if "diameter_px" in metrics_px and metrics_px["diameter_px"]:
        metrics_mm["diameter_mm"] = float(metrics_px["diameter_px"] * mm_per_px)
    
    # Convert area if available
    if "area_px" in metrics_px and metrics_px["area_px"]:
        metrics_mm["area_mm2"] = float(metrics_px["area_px"] * (mm_per_px ** 2))
    
    # Keep center coordinates in pixels (no conversion needed)
    if "center_xy" in metrics_px:
        metrics_mm["center_xy"] = metrics_px["center_xy"]
    
    return metrics_mm
