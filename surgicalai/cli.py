# SPDX-License-Identifier: Apache-2.0
"""SurgicalAI command line interface."""

from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import typer

from surgicalai import demo as demo_mod
from surgicalai.training import train as train_mod
from surgicalai.training import evaluate as eval_mod

app = typer.Typer(help="SurgicalAI utilities")


@app.command()
def demo(
    input: Path = typer.Option(..., "--input", help="Input sample directory"),
    out: Path = typer.Option(Path("runs/demo"), "--out", help="Output directory"),
    cpu: bool = typer.Option(True, "--cpu", help="Force CPU mode"),
    offline_llm: bool = typer.Option(
        True, "--offline-llm", help="Disable network calls"
    ),
) -> None:
    """Run demo pipeline."""
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if offline_llm:
        os.environ["SURGICALAI_OFFLINE_LLM"] = "1"
    demo_mod.run(out, input_dir=input, with_llm=not offline_llm)


@app.command()
def train(data: Path = typer.Option(Path("data/lesions_sample"), "--data")) -> None:
    """Train toy model on synthetic data."""
    train_mod.train(data)


@app.command()
def evaluate(checkpoint: Path = typer.Option(Path("toy.pt"), "--checkpoint")) -> None:
    """Evaluate toy model."""
    eval_mod.evaluate(checkpoint)


@app.command()
def api(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run FastAPI server."""
    os.environ.setdefault("SURGICALAI_OFFLINE_LLM", "1")
    import uvicorn

    uvicorn.run("surgicalai.api:app", host=host, port=port)


@app.command()
def ui() -> None:
    """Launch Gradio UI."""
    os.environ.setdefault("SURGICALAI_OFFLINE_LLM", "1")
    from surgicalai import ui as ui_mod

    ui_mod.launch()


@app.command()
def package() -> None:
    """Build standalone executable via PyInstaller."""
    import PyInstaller.__main__

    spec = Path(__file__).resolve().parent.parent / "surgicalai.spec"
    PyInstaller.__main__.run([str(spec)])


if __name__ == "__main__":  # pragma: no cover
    app()
