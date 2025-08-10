from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
from skimage import color, filters, measure, morphology, exposure

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
    # Simple YCrCb threshold → skin-ish area; Tier‑0 heuristic
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=256)
    mask = morphology.remove_small_objects(mask, min_size=256)
    return mask.astype(np.uint8)*255

def _lesion_candidate(rgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    # Illumination correction + Otsu on V channel within skin
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[...,2]
    v_eq = exposure.equalize_adapthist(v, clip_limit=0.01)
    v_uint8 = (v_eq*255).astype(np.uint8)

    v_skin = cv2.bitwise_and(v_uint8, v_uint8, mask=skin)
    thr = filters.threshold_otsu(v_skin[v_skin>0]) if (v_skin>0).any() else 128
    lesion = (v_skin < thr).astype(np.uint8)*255  # darker than surround
    lesion = cv2.medianBlur(lesion,5)
    lesion = morphology.remove_small_objects(lesion.astype(bool), 200).astype(np.uint8)*255
    lesion = morphology.binary_closing(lesion, morphology.disk(3)).astype(np.uint8)*255
    return lesion

def _largest_blob(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask>0, connectivity=2)
    if labels.max() == 0: 
        return np.zeros_like(mask)
    regions = measure.regionprops(labels)
    regions.sort(key=lambda r: r.area, reverse=True)
    lb = (labels == regions[0].label).astype(np.uint8)*255
    return lb

def _asymmetry_score(blob: np.ndarray) -> float:
    ys, xs = np.where(blob>0)
    if len(xs) < 20: return 0.0
    coords = np.column_stack([xs, ys]).astype(np.float32)
    mean = coords.mean(0)
    coords0 = coords - mean
    cov = np.cov(coords0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    rot = np.array([[major[0], -major[1]], [major[1], major[0]]])
    aligned = coords0 @ rot
    left = (aligned[:,0] < 0).sum()
    right = (aligned[:,0] >= 0).sum()
    area_diff = abs(left - right) / max(1, (left + right))
    return float(area_diff)

def _border_irregularity(blob: np.ndarray) -> float:
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if area <= 1: return 0.0
    return float((perim**2) / (4*np.pi*area))

def _color_variegation(rgb: np.ndarray, blob: np.ndarray) -> float:
    # Count distinct hue clusters within blob mask (k=1..4) based on variance drop
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[...,0][blob>0].reshape(-1,1).astype(np.float32)
    if len(H) < 50: return 0.0
    from sklearn.cluster import KMeans
    best_k, best_inertia = 1, None
    for k in range(1,5):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(H)
        if best_inertia is None or km.inertia_ < best_inertia*0.7:
            best_k, best_inertia = k, km.inertia_
    return float(best_k - 1)  # 0..3 extra clusters

def _diameter_px(blob: np.ndarray) -> float:
    ys, xs = np.where(blob>0)
    if len(xs) < 2: return 0.0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return float(max(x1 - x0, y1 - y0))

def _elevation_satellite(blob: np.ndarray) -> float:
    # Proxy: small secondary blobs count near the main lesion
    dil = cv2.dilate(blob, np.ones((7,7), np.uint8), iterations=1)
    edge = cv2.Canny(dil, 50, 150)
    satellites = (edge>0).sum() / 1000.0
    return float(min(satellites, 5.0))

def compute_abcde(rgb: np.ndarray) -> tuple[ABCDE, np.ndarray]:
    skin = _skin_mask(rgb)
    lesion = _lesion_candidate(rgb, skin)
    blob = _largest_blob(lesion)
    A = _asymmetry_score(blob)
    B = _border_irregularity(blob)
    C = _color_variegation(rgb, blob)
    D = _diameter_px(blob)
    E = _elevation_satellite(blob)
    return ABCDE(A,B,C,D,E), blob

def save_overlay(rgb: np.ndarray, blob: np.ndarray, out_path: Path):
    overlay = rgb.copy()
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, cnts, -1, (255,0,0), 2)
        # crude heat: distance transform
        dist = cv2.distanceTransform((blob>0).astype(np.uint8), cv2.DIST_L2, 5)
        dist = (255*dist/dist.max()) if dist.max()>0 else dist
        heat = cv2.applyColorMap(dist.astype(np.uint8), cv2.COLORMAP_JET)
        mask3 = np.dstack([blob>0]*3)
        overlay = np.where(mask3, (0.6*overlay+0.4*heat).astype(np.uint8), overlay)
    Image.fromarray(overlay).save(out_path)

def dump_metrics(abcde: ABCDE, extras: dict, out_json: Path):
    payload = {
        "asymmetry": abcde.asymmetry,
        "border_irregularity": abcde.border_irregularity,
        "color_variegation": abcde.color_variegation,
        "diameter_px": abcde.diameter_px,
        "elevation_satellite": abcde.elevation_satellite,
        **extras
    }
    out_json.write_text(json.dumps(payload, indent=2))
