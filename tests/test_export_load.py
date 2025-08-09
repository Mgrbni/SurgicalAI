from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner
import torch
import onnxruntime as ort

from surgicalai.cli import app


def _create_dataset(root: Path) -> None:
    for cls in ["benign", "melanoma"]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1):
            img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
            img.save(d / f"{i}.jpg")


def test_export_load(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "lesions"
    _create_dataset(data_root)
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            app,
            [
                "train",
                "--data-source",
                "folder",
                "--root",
                str(data_root),
                "--epochs",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        traced = Path("models/resnet50_traced.pt")
        onnx = Path("models/resnet50.onnx")
        assert traced.exists() and onnx.exists()
        torch.jit.load(traced)
        ort.InferenceSession(str(onnx))
