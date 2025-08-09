from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_cli_demo(tmp_path: Path) -> None:
    out = tmp_path / "demo"
    cmd = ["surgicalai", "demo", "--out", str(out)]
    if os.getenv("OPENAI_API_KEY"):
        cmd.append("--with-llm")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
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
        assert (out / name).exists()
    if os.getenv("OPENAI_API_KEY"):
        assert (out / "narrative.json").exists()
