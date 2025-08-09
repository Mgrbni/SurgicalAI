from __future__ import annotations

from pathlib import Path

from surgicalai.demo import run as demo_run


def test_synthetic_assets(tmp_path: Path) -> None:
    out = tmp_path / "case"
    demo_run(out)
    assert (out / "mesh.npz").exists()
    assert (out / "anatomy" / "langer_lines.npy").exists()
    assert (out / "anatomy" / "arteries.json").exists()
    assert (out / "anatomy" / "nerves.json").exists()
