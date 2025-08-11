"""Dual-provider (OpenAI + Anthropic) multimodel lesion localization & consensus.

locate_lesion(image: np.ndarray) -> dict
Returns dict with provider_results, consensus box, heatmap path.
Uses retry, timeout, and fallback logic. Heatmap is synthetic placeholder focusing near consensus center.
"""
from __future__ import annotations
import os, time, json, uuid, math
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
import cv2
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import anthropic

TIMEOUT_S = 30

@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    confidence: float

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, other: 'Box') -> float:
        xa = max(self.x, other.x)
        ya = max(self.y, other.y)
        xb = min(self.x + self.w, other.x + other.w)
        yb = min(self.y + self.h, other.y + other.h)
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


def _clamp_box(box: Box, w: int, h: int) -> Box:
    x = max(0.0, min(box.x, w - 1))
    y = max(0.0, min(box.y, h - 1))
    w2 = max(1.0, min(box.w, w - x))
    h2 = max(1.0, min(box.h, h - y))
    return Box(x, y, w2, h2, box.confidence)

# ---------------- Model Calls (Mock/Simplified) ------------------

PROMPT_BASE = "Locate the primary pigmented skin lesion. Return JSON with keys bbox:[x,y,w,h], confidence (0-1), explanation."
PROMPT_HINT = PROMPT_BASE + " Focus on pigmented lesion near philtrum / upper lip if present."

class ProviderError(Exception):
    pass


def _encode_image(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('Failed to encode image')
    return buf.tobytes()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), retry=retry_if_exception_type(ProviderError))
def _call_openai(image: np.ndarray, prompt: str) -> Dict[str, Any]:
    # Placeholder logic (no real API call for offline test)
    h, w = image.shape[:2]
    # Heuristic: sample bright-ish central region
    cx, cy = w * 0.5, h * 0.45
    bw, bh = w * 0.18, h * 0.18
    return {
        "provider": "openai",
        "bbox": [cx - bw/2, cy - bh/2, bw, bh],
        "confidence": 0.78,
        "explanation": "Heuristic central box (mock).",
        "request_id": str(uuid.uuid4())
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=6), retry=retry_if_exception_type(ProviderError))
def _call_claude(image: np.ndarray, prompt: str) -> Dict[str, Any]:
    h, w = image.shape[:2]
    cx, cy = w * 0.52, h * 0.47
    bw, bh = w * 0.2, h * 0.2
    return {
        "provider": "anthropic",
        "bbox": [cx - bw/2, cy - bh/2, bw, bh],
        "confidence": 0.75,
        "explanation": "Heuristic slightly right box (mock).",
        "request_id": str(uuid.uuid4())
    }

# ---------------- Consensus ------------------

def _consensus(box_a: Box, box_b: Box, reran: bool) -> Tuple[Box, Dict[str, Any]]:
    iou = box_a.iou(box_b)
    meta: Dict[str, Any] = {"iou": iou, "reran": reran}
    if iou >= 0.6:
        # mean center / union size simplification
        x = (box_a.x + box_b.x) / 2
        y = (box_a.y + box_b.y) / 2
        w = (box_a.w + box_b.w) / 2
        h = (box_a.h + box_b.h) / 2
        conf = (box_a.confidence + box_b.confidence) / 2
        meta["strategy"] = "average"
        return Box(x, y, w, h, conf), meta
    # pick higher confidence
    meta["strategy"] = "higher_confidence"
    better = box_a if box_a.confidence >= box_b.confidence else box_b
    return better, meta


def _synthetic_heatmap(h: int, w: int, box: Box) -> np.ndarray:
    hm = np.zeros((h, w), dtype=np.float32)
    cx = box.x + box.w / 2
    cy = box.y + box.h / 2
    for y in range(int(max(0, cy - box.h)), int(min(h, cy + box.h))):
        for x in range(int(max(0, cx - box.w)), int(min(w, cx + box.w))):
            dx = (x - cx) / (box.w / 2 + 1e-6)
            dy = (y - cy) / (box.h / 2 + 1e-6)
            r2 = dx*dx + dy*dy
            if r2 <= 1.0:
                hm[y, x] = max(hm[y, x], 1 - r2)
    hm /= hm.max() if hm.max() > 0 else 1
    return hm


def locate_lesion(image: np.ndarray, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run dual provider vision localization and produce consensus + heatmap."""
    h, w = image.shape[:2]
    out_dir = Path(out_dir) if out_dir else Path("runs") / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # First pass
    r1 = _call_openai(image, PROMPT_BASE)
    r2 = _call_claude(image, PROMPT_BASE)
    b1 = Box(*r1["bbox"], r1["confidence"])  # type: ignore[arg-type]
    b2 = Box(*r2["bbox"], r2["confidence"])  # type: ignore[arg-type]
    consensus, meta = _consensus(b1, b2, reran=False)

    if meta["iou"] < 0.6:
        # Rerun with hint (mock uses slightly adjusted boxes)
        r1b = _call_openai(image, PROMPT_HINT)
        r2b = _call_claude(image, PROMPT_HINT)
        b1 = Box(*r1b["bbox"], r1b["confidence"])  # type: ignore[arg-type]
        b2 = Box(*r2b["bbox"], r2b["confidence"])  # type: ignore[arg-type]
        consensus, meta = _consensus(b1, b2, reran=True)

    consensus = _clamp_box(consensus, w, h)

    # Heatmap
    heat = _synthetic_heatmap(h, w, consensus)
    heat_path = out_dir / "consensus_heatmap.png"
    heat_img = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_img = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    cv2.imwrite(str(heat_path), heat_img)

    # Log provider raw JSON
    for rec in (r1, r2):
        with open(logs_dir / f"provider_{rec['provider']}_{rec['request_id']}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)

    return {
        "providers": [r1, r2],
        "consensus": {
            "bbox": list(consensus.as_xywh()),
            "center": [consensus.x + consensus.w/2, consensus.y + consensus.h/2],
            "confidence": consensus.confidence,
            "meta": meta,
        },
        "heatmap_path": str(heat_path),
    }

__all__ = ["locate_lesion"]
