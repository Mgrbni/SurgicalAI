"""Inference + artifact pipeline orchestration (demo).

Provides process_image which:
- Runs multi-class inference (softmax with temperature calibration)
- Computes oncology gate decision
- Persists metrics.json with probs, topk, calibration meta, gate
- Ensures top-3 Grad-CAM heatmaps exist (placeholders)

NOT FOR CLINICAL USE.
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any
from .model_infer import run_inference
from .rules import compute_gate

def process_image(image_path: Path, run_dir: Path, debug: bool = False) -> Dict[str, Any]:
    result = run_inference(image_path, run_dir, topk=3)
    probs = {k: round(v, 4) for k, v in result.probs.items()}
    gate = compute_gate(probs, result.topk)

    metrics = {
        "run_id": run_dir.name,
        "probs": probs,
        "topk": result.topk,
        "temperature": result.temperature,
        "class_weights": result.class_weights,
        "class_bias": result.class_bias,
        "gate": gate,
    }
    if debug:
        metrics["debug"] = {
            "logits": result.logits,
            "calibrated_logits": result.calibrated_logits,
        }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Collect heatmaps for top-3
    heatmaps = {}
    for lbl,_ in result.topk:
        hp = run_dir / f"heatmap_{lbl}.png"
        if hp.exists():
            heatmaps[lbl] = hp.name

    return {
        "metrics": metrics,
        "heatmaps": heatmaps,
    }
