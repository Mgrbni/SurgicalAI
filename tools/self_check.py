from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    out = Path("outputs/selfcheck")
    subprocess.run(["surgicalai", "demo", "--out", str(out)], check=True)
    required = [
        "mesh.npz",
        "heatmap_vertex.npy",
        "lesion_probs.json",
        "flap_plan.json",
        "contraindications.json",
        "surgical_plan.glb",
        "overview.png",
        "report.pdf",
    ]
    for name in required:
        p = out / name
        if not p.exists() or p.stat().st_size == 0:
            return 1
    model_path = Path("models/resnet50_best.pt")
    if model_path.exists():
        import torch

        torch.load(model_path, map_location="cpu")
    if (out / "narrative.json").exists() and (
        out / "narrative.json"
    ).stat().st_size == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
