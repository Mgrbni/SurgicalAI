from __future__ import annotations

import subprocess


def test_cli(synthetic_mesh, tmp_path):
    out_dir = tmp_path / "out"
    cmd = [
        "python",
        "surgicalai_cli.py",
        "run",
        "--input",
        str(synthetic_mesh),
        "--output-dir",
        str(out_dir),
        "--device",
        "cpu",
        "--report",
        "html",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out_dir / "features.npz").exists()
    assert "pipeline complete" in proc.stdout
