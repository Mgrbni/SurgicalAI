"""Command line interface for SurgicalAI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console

from surgicalai import (
    __version__,
    mesh_parser,
    lesion_detection,
    flap_designer,
    biomech_validator,
    utils,
    viz,
)

app = typer.Typer(add_completion=False)
console = Console()


@app.callback()
def main() -> None:
    """SurgicalAI command line interface."""


@app.command("run")
def run(
    input: Path = typer.Option(..., help="Path to Polycam mesh"),
    output_dir: Path = typer.Option(..., help="Directory to store artifacts"),
    device: str = typer.Option("cpu", help="Computation device"),
    weights: Optional[Path] = typer.Option(None, help="Model weights"),
    report: str = typer.Option("html", help="Report format"),
) -> None:
    """Run the full SurgicalAI pipeline on *input* mesh."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        mesh_data = mesh_parser.load_mesh(input)
        np.savez(output_dir / "features.npz", coordinates=mesh_data["coordinates"])
        utils.save_json(mesh_data["landmarks"], output_dir / "landmarks.json")

        lesion = lesion_detection.detect(mesh_data, device=device, weights=str(weights) if weights else None)
        utils.save_json(lesion["class_probs"], output_dir / "lesion_probs.json")

        heatmap_path = output_dir / "heatmap_overlay.png"
        viz.overlay_heatmap(mesh_data, lesion["heatmap"], heatmap_path)

        flap_plan = flap_designer.design(mesh_data, mesh_data["coordinates"].mean(axis=0))
        utils.save_json(flap_plan, output_dir / "flap_plan.json")

        flap_path = output_dir / "flap_overlay.png"
        viz.overlay_flap(mesh_data, flap_plan, flap_path)

        validator = biomech_validator.validate(flap_plan)
        utils.save_json(validator, output_dir / "validator.json")

        report_path = output_dir / "report.html"
        viz.export_report({"heatmap": heatmap_path, "flap": flap_path}, report_path)

        console.print(f"SurgicalAI: build v{__version__} – pipeline complete ({report_path})")
    except Exception as exc:  # pragma: no cover - error path
        console.print(f"[red]Error: {exc}")
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
