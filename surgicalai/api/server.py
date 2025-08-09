"""FastAPI server exposing pipeline."""
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
from pathlib import Path
import uuid
from ..pipeline.integration import Pipeline
from ..planning.flap_designer import design_flap
from ..risk.contraindications import assess_risk
from ..risk.biomech_validator import estimate_strain, validate_strain
from ..viz.overlays import overlay_heatmap
from ..viz.export import save_image
from ..config import CONFIG

app = FastAPI(title="surgicalai", version="0.1")
pipeline = Pipeline()

class AnalyzeResponse(BaseModel):
    probs: dict
    run_id: str

@app.post('/ingest')
async def ingest(scan: UploadFile = File(...)):
    run_id = uuid.uuid4().hex[:8]
    out_dir = CONFIG.outputs / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_path = out_dir / scan.filename
    with open(scan_path, 'wb') as f:
        f.write(await scan.read())
    return {'run_id': run_id, 'scan_path': str(scan_path)}

@app.post('/analyze', response_model=AnalyzeResponse)
async def analyze(run_id: str, photo: UploadFile | None = None):
    out_dir = CONFIG.outputs / run_id
    scan_files = list(out_dir.glob('*.obj')) + list(out_dir.glob('*.ply')) + list(out_dir.glob('*.glb'))
    if not scan_files:
        return JSONResponse({'error':'no scan'}, status_code=400)
    scan_path = scan_files[0]
    photo_path = None
    if photo:
        photo_path = out_dir / photo.filename
        with open(photo_path,'wb') as f:
            f.write(await photo.read())
    result = pipeline.run(scan_path, photo_path, run_id=run_id)
    return AnalyzeResponse(probs=result['probs'], run_id=run_id)

@app.post('/plan')
async def plan(run_id: str):
    out_dir = CONFIG.outputs / run_id
    summary_path = out_dir / 'summary.json'
    if not summary_path.exists():
        return JSONResponse({'error':'run analyze first'}, status_code=400)
    data = pipeline.run(out_dir / 'face_mock.obj', run_id=run_id)
    flap = design_flap(data['vertices'], data['heat'])
    risk = assess_risk(data['vertices'][0])
    strain = estimate_strain(data['vertices'], data['vertices'].astype(float), np.array(flap['vector']))
    strain_res = validate_strain(strain)
    return {'flap': flap, 'risk': risk, 'strain': strain_res}

@app.get('/visualize')
async def visualize(run_id: str):
    out_dir = CONFIG.outputs / run_id
    summary = out_dir / 'summary.json'
    if not summary.exists():
        return JSONResponse({'error':'not found'}, status_code=404)
    img_path = out_dir / 'overlay.png'
    if img_path.exists():
        return FileResponse(img_path)
    return JSONResponse({'error':'no image'}, status_code=404)

@app.get('/report')
async def report(run_id: str):
    out_dir = CONFIG.outputs / run_id
    report = out_dir / 'report.html'
    if not report.exists():
        return JSONResponse({'error':'not found'}, status_code=404)
    return FileResponse(report)
