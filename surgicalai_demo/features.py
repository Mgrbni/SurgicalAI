from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
from skimage import filters, measure, morphology, exposure


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
    return mask.astype(np.uint8) * 255


def _lesion_candidate(rgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """Initial dark-object proposal inside skin using HSV-V Otsu + cleanup."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    v_eq = exposure.equalize_adapthist(v, clip_limit=0.01)
    v_u8 = (v_eq * 255).astype(np.uint8)
    v_skin = cv2.bitwise_and(v_u8, v_u8, mask=skin)

    thr = filters.threshold_otsu(v_skin[v_skin > 0]) if (v_skin > 0).any() else 128
    lesion = (v_skin < thr).astype(np.uint8) * 255  # darker than surround
    lesion = cv2.medianBlur(lesion, 5)
    lesion = (
        morphology.remove_small_objects(lesion.astype(bool), 200).astype(np.uint8) * 255
    )
    lesion = (
        morphology.binary_closing(lesion, morphology.disk(3)).astype(np.uint8) * 255
    )
    return lesion


def _largest_blob(
    mask: np.ndarray, min_frac: float = 0.001, max_frac: float = 0.3
) -> np.ndarray:
    """Pick a sane lesion: largest component with area in [min,max] of image."""
    H, W = mask.shape[:2]
    labels = measure.label(mask > 0, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(mask)
    regions = [r for r in measure.regionprops(labels)]
    regions.sort(key=lambda r: r.area, reverse=True)
    area_min, area_max = min_frac * H * W, max_frac * H * W
    for r in regions:
        if area_min <= r.area <= area_max:
            return (labels == r.label).astype(np.uint8) * 255
    # fallback to biggest but clamp
    r = regions[0]
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
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    rect = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = np.zeros(blob.shape, np.uint8) + cv2.GC_PR_BGD
    mask[blob > 0] = cv2.GC_PR_FGD

    try:
        cv2.grabCut(rgb, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return blob

    refined = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )
    refined = cv2.medianBlur(refined, 3)
    refined = (
        morphology.remove_small_objects(refined.astype(bool), 200).astype(np.uint8)
        * 255
    )
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
    return float((perim**2) / (4 * np.pi * area))


def _color_variegation(rgb: np.ndarray, blob: np.ndarray) -> float:
    # Count distinct hue clusters within blob mask (k=1..4) based on variance drop
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H = hsv[..., 0][blob > 0].reshape(-1, 1).astype(np.float32)
    if len(H) < 50:
        return 0.0
    from sklearn.cluster import KMeans

    best_k, best_inertia = 1, None
    for k in range(1, 5):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(H)
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
    # Proxy: small secondary blobs count near the main lesion
    dil = cv2.dilate(blob, np.ones((7, 7), np.uint8), iterations=1)
    edge = cv2.Canny(dil, 50, 150)
    satellites = (edge > 0).sum() / 1000.0
    return float(min(satellites, 5.0))


def compute_abcde(rgb: np.ndarray) -> tuple[ABCDE, np.ndarray]:
    skin = _skin_mask(rgb)
    lesion0 = _lesion_candidate(rgb, skin)
    blob = _largest_blob(lesion0)
    blob = _refine_with_grabcut(rgb, blob)  # ← new: tighten mask

    A = _asymmetry_score(blob)
    B = _border_irregularity(blob)
    C = _color_variegation(rgb, blob)
    D = _diameter_px(blob)
    E = _elevation_satellite(blob)
    return ABCDE(A, B, C, D, E), blob


def _attention_map(rgb: np.ndarray, blob: np.ndarray) -> np.ndarray:
    """Attention = inside-distance + local darkness contrast; normalized to [0,1]."""
    H, W = blob.shape
    inside = cv2.distanceTransform((blob > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if inside.max() > 0:
        inside = inside / (inside.max() + 1e-6)
    else:
        inside = np.zeros_like(inside, dtype=np.float32)

    # darkness contrast vs local mean
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    L_blur = cv2.GaussianBlur(L, (0, 0), 3)
    contrast = L_blur - L  # darker → positive
    # clamp & normalize only within a soft band around lesion
    band = cv2.dilate(
        (blob > 0).astype(np.uint8), np.ones((21, 21), np.uint8), iterations=1
    )
    c_vals = contrast[band > 0]
    if c_vals.size > 0:
        lo, hi = np.percentile(c_vals, [5, 95])
        contrast = np.clip((contrast - lo) / (hi - lo + 1e-6), 0, 1)
    else:
        contrast = np.zeros_like(inside, dtype=np.float32)

    att = 0.6 * inside + 0.4 * contrast
    att = att * (band > 0)  # never leak outside
    att = cv2.GaussianBlur(att, (0, 0), 2)
    if att.max() > 0:
        att = att / att.max()
    return att.astype(np.float32)


def save_overlay(rgb: np.ndarray, blob: np.ndarray, out_path: Path):
    vis = rgb.copy()
    # draw contour
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(vis, cnts, -1, (255, 0, 0), 2)

    # attention heatmap restricted to lesion band
    att = _attention_map(rgb, blob)
    heat = cv2.applyColorMap((att * 255).astype(np.uint8), cv2.COLORMAP_JET)
    band = (att > 0).astype(np.uint8)
    band3 = np.dstack([band] * 3)
    blended = np.where(band3 > 0, (0.65 * vis + 0.35 * heat).astype(np.uint8), vis)

    # small feather to avoid hard edges
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.dilate((blob > 0).astype(np.uint8), kernel, 1) - (blob > 0).astype(
        np.uint8
    )
    blended[edge > 0] = (
        0.5 * blended[edge > 0] + 0.5 * np.array([255, 255, 255])
    ).astype(np.uint8)

    Image.fromarray(blended).save(out_path)


def dump_metrics(abcde: ABCDE, extras: dict, out_json: Path):
    payload = {
        "asymmetry": abcde.asymmetry,
        "border_irregularity": abcde.border_irregularity,
        "color_variegation": abcde.color_variegation,
        "diameter_px": abcde.diameter_px,
        "elevation_satellite": abcde.elevation_satellite,
        **extras,
    }
    out_json.write_text(json.dumps(payload, indent=2))
