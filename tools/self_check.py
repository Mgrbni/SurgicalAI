from __future__ import annotations

from pathlib import Path
import sys

from surgicalai.demo import run


def main() -> int:
    out = Path("outputs/selfcheck")
    run(out)
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
    if (out / "narrative.json").exists() and (out / "narrative.json").stat().st_size == 0:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
