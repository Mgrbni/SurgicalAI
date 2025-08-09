"""CLI entrypoints."""
from __future__ import annotations
import typer
from pathlib import Path
from .pipeline.integration import Pipeline
from .planning.flap_designer import design_flap
from .risk.contraindications import assess_risk
from .risk.biomech_validator import estimate_strain, validate_strain
from .viz.overlays import overlay_heatmap
from .viz.export import save_image
from .config import CONFIG
from PIL import Image
import numpy as np
from .report.generator import generate_report
from .utils.io import ensure_dir

app = typer.Typer()
pipe = Pipeline()

@ app.command()
def ingest(scan: Path, run_id: str = 'cli'):
    out = CONFIG.outputs / run_id
    ensure_dir(out)
    dest = out / scan.name
    dest.write_bytes(Path(scan).read_bytes())
    typer.echo(f"Saved scan to {dest}")

@ app.command()
def analyze(run_id: str = 'cli', photo: Path | None = None):
    scan_files = list((CONFIG.outputs / run_id).glob('*'))
    scan_path = next((f for f in scan_files if f.suffix in ['.obj','.ply','.glb']), None)
    if not scan_path:
        raise typer.Exit('Scan not found')
    res = pipe.run(scan_path, photo, run_id=run_id)
    typer.echo(res['probs'])

@ app.command()
def plan(run_id: str = 'cli'):
    out_dir = CONFIG.outputs / run_id
    res = pipe.run(out_dir / 'face_mock.obj', run_id=run_id)
    flap = design_flap(res['vertices'], res['heat'])
    typer.echo(flap)

@ app.command()
def visualize(run_id: str = 'cli'):
    out_dir = CONFIG.outputs / run_id
    typer.echo(f"See results in {out_dir}")

@ app.command()
def demo(all: bool = typer.Option(False, "--all", help="Run full demo")):
    scan = CONFIG.data / 'samples/face_mock.obj'
    # generate placeholder photo on the fly
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    res = pipe.run(scan, run_id='demo')
    flap = design_flap(res['vertices'], res['heat'])
    risk = assess_risk(res['vertices'][0])
    strain = validate_strain(estimate_strain(res['vertices'], res['vertices'], np.array(flap['vector'])))
    heat_img = overlay_heatmap(img, res['heatmap'])
    out_dir = CONFIG.outputs / res['run_id']
    save_image(heat_img, out_dir / 'overlay.png')
    generate_report(res['run_id'], {'probs': res['probs'], 'flap': flap, 'risk': risk, 'strain': strain}, out_dir)
    typer.echo(f"Results stored in {out_dir}")
