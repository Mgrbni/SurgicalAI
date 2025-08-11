"""Advanced API mirroring simple analyze endpoint with enriched response.

Currently delegates to the same pipeline logic. This provides a placeholder
for future advanced parameters (e.g., model selection, extra imaging) while
maintaining parity with /api/analyze in api_simple.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
app = FastAPI(title="SurgicalAI Advanced API")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.post("/api/advanced/analyze")
async def advanced_analyze(
	file: UploadFile = File(...),
	payload: str = Form(...)
):
	if not file.content_type or not file.content_type.startswith('image/'):
		raise HTTPException(status_code=400, detail="File must be an image")
	try:
		data = json.loads(payload)
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid JSON payload")

	try:
		from surgicalai_demo.pipeline import run_demo
		from datetime import datetime
		import uuid

		run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4]
		run_dir = Path("runs") / run_id
		run_dir.mkdir(parents=True, exist_ok=True)

		img_path = run_dir / "input.jpg"
		with open(img_path, 'wb') as f:
			f.write(await file.read())

		summary = run_demo(None, img_path, run_dir, ask=False)
		paths = summary.get("paths", {})
		fusion = summary.get("fusion", {})
		observer = summary.get("observer", {})
		probs = summary.get("probs", {})
		top3 = (fusion.get("top3") or sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3])

		artifacts = {
			"overlay_full": f"/api/artifact/{run_id}/overlay_full.png" if paths.get("overlay_full_png") else None,
			"overlay_zoom": f"/api/artifact/{run_id}/overlay_zoom.png" if paths.get("overlay_zoom_png") else None,
			"pdf": f"/api/artifact/{run_id}/report.pdf",
		}
		artifacts = {k: v for k, v in artifacts.items() if v}

		return {
			"ok": True,
			"run_id": run_id,
			"probs": probs,
			"top3": top3,
			"observer": {
				"primary_pattern": observer.get("primary_pattern"),
				"likelihoods": observer.get("likelihoods", {}),
				"descriptors": observer.get("descriptors", []),
				"abcd_estimate": observer.get("abcd_estimate", {}),
				"recommendation": observer.get("recommendation"),
			} if observer else {},
			"fusion": {
				"top3": fusion.get("top3", []),
				"notes": fusion.get("fusion_notes", fusion.get("notes", "")),
			} if fusion else {},
			"artifacts": artifacts,
		}
	except Exception as e:
		logger.exception("Advanced analysis failed")
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/artifact/{path:path}")
async def get_artifact(path: str):
	artifact_path = Path("runs") / path
	if not artifact_path.exists():
		raise HTTPException(status_code=404, detail="Artifact not found")
	return FileResponse(artifact_path)
