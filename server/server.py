import os
import pathlib
import json
import logging
from datetime import datetime, timezone
import uuid
import hashlib
import tempfile
import threading
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from server.settings import SETTINGS
from server.schemas import LesionReport, LesionRequest, FacialSubunitEnum, AnalysisOutput
from server.rate_limit import retry_policy
from server.llm import LLMRouter, OpenAIClient, AnthropicClient, LLMError, LLMValidationError
from server.usage import log_usage

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
load_dotenv(ROOT / ".env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM Router
def create_llm_router() -> LLMRouter:
    """Create and configure the LLM router"""
    try:
        openai_config = None
        if SETTINGS.openai_api_key:
            openai_config = {
                "api_key": SETTINGS.openai_api_key,
                "model": SETTINGS.openai_model,
                "timeout_seconds": SETTINGS.timeout_seconds,
                "project": SETTINGS.openai_project or None,
                "organization": SETTINGS.openai_org or None
            }
        
        anthropic_config = None
        if SETTINGS.anthropic_api_key:
            anthropic_config = {
                "api_key": SETTINGS.anthropic_api_key,
                "model": SETTINGS.anthropic_model,
                "timeout_seconds": SETTINGS.timeout_seconds
            }
        
        router = LLMRouter.from_config(
            provider=SETTINGS.provider,
            openai_config=openai_config,
            anthropic_config=anthropic_config
        )
        
        logger.info(f"LLM Router initialized: {router.active_provider}/{router.active_model}")
        return router
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM router: {e}")
        raise RuntimeError(f"LLM configuration error: {e}")

# Global LLM router instance
llm_router = create_llm_router()

# Curated facial subunit mapping
FACIAL_SUBUNIT_LABELS = {
    "forehead": "Forehead",
    "temple": "Temple", 
    "eyebrow": "Eyebrow",
    "eyelid_upper": "Upper Eyelid",
    "eyelid_lower": "Lower Eyelid",
    "canthus_medial": "Medial Canthus",
    "canthus_lateral": "Lateral Canthus",
    "nose_dorsum": "Nasal Dorsum",
    "nose_tip": "Nasal Tip",
    "nose_ala": "Nasal Ala",
    "cheek_medial": "Medial Cheek",
    "cheek_lateral": "Lateral Cheek",
    "zygomatic": "Zygomatic (Malar)",
    "lip_upper": "Upper Lip",
    "lip_lower": "Lower Lip",
    "philtrum": "Philtrum",
    "chin": "Chin",
    "jawline": "Jawline",
    "ear_helix": "Ear (Helix)",
    "ear_lobe": "Ear (Lobe)",
    "parotid_region": "Parotid Region",
    "neck_anterior": "Anterior Neck",
}

app = FastAPI(title="SurgicalAI API")

# CORS (be permissive for demo; lock down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceReq(BaseModel):
    prompt: str
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    json_schema: bool = False  # set true for lesion/flap JSON

@app.get("/healthz")
def healthz():
    """Enhanced health check with provider info"""
    try:
        return {
            "ok": True,
            "provider": llm_router.active_provider,
            "model": llm_router.active_model,
            "time": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "time": datetime.now(timezone.utc).isoformat()
        }

@app.get("/api/health")
def health():
    """Legacy health endpoint"""
    return {"ok": True, "model": SETTINGS.model}

@app.get("/api/facial-subunits")
def get_facial_subunits():
    """Return curated facial subunit options."""
    subunits = []
    for value, label in FACIAL_SUBUNIT_LABELS.items():
        subunits.append({"value": value, "label": label})
    return {"subunits": subunits}

@app.get("/api/last-usage")
async def get_last_usage(limit: int = 10):
    """Get recent usage logs"""
    try:
        # Mock implementation for now - in production this would read from a database
        import glob
        import os
        from pathlib import Path
        
        runs = []
        if RUNS_ROOT.exists():
            run_dirs = sorted(RUNS_ROOT.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]
            
            for run_dir in run_dirs:
                if run_dir.is_dir():
                    try:
                        # Try to read analysis results
                        analysis_file = run_dir / 'analysis.json'
                        metrics_file = run_dir / 'metrics.json'
                        
                        entry = {
                            "id": run_dir.name,
                            "request_id": run_dir.name,
                            "timestamp": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                            "time": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                            "success": True,
                            "subunit": "unknown",
                            "primary_diagnosis": "unknown",
                            "provider": "unknown",
                            "model": "unknown"
                        }
                        
                        if analysis_file.exists():
                            try:
                                import json
                                with open(analysis_file, 'r') as f:
                                    analysis = json.load(f)
                                entry.update({
                                    "primary_diagnosis": analysis.get("primary_dx", "unknown"),
                                    "subunit": analysis.get("site", "unknown")
                                })
                            except Exception:
                                pass
                        
                        if metrics_file.exists():
                            try:
                                import json
                                with open(metrics_file, 'r') as f:
                                    metrics = json.load(f)
                                entry.update({
                                    "provider": metrics.get("analysis", {}).get("provider", "unknown"),
                                    "model": metrics.get("analysis", {}).get("model", "unknown"),
                                    "subunit": metrics.get("input", {}).get("subunit", entry["subunit"])
                                })
                            except Exception:
                                pass
                        
                        runs.append(entry)
                    except Exception:
                        continue
        
        return runs
    except Exception as e:
        logger.warning(f"Failed to get usage logs: {e}")
        return []

def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

def _write_atomic(path: pathlib.Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_name = tmp.name
    os.replace(temp_name, path)

def _hash_file(path: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def _run_housekeeping(retention_hours: int = 24):
    while True:
        try:
            cutoff = datetime.now(timezone.utc).timestamp() - retention_hours * 3600
            if RUNS_ROOT.exists():
                for d in RUNS_ROOT.iterdir():
                    if d.is_dir():
                        try:
                            # parse run id datetime prefix
                            prefix = d.name.split('-')[0] + '-' + d.name.split('-')[1]
                            # fallback to mtime if parse fails
                            ts = d.stat().st_mtime
                            if ts < cutoff:
                                for rootp, dirs, files in os.walk(d, topdown=False):
                                    for name in files:
                                        try: os.remove(pathlib.Path(rootp)/name)
                                        except: pass
                                    for name in dirs:
                                        try: os.rmdir(pathlib.Path(rootp)/name)
                                        except: pass
                                try: os.rmdir(d)
                                except: pass
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Housekeeping error: {e}")
        time.sleep(3600)

@app.on_event("startup")
def _start_housekeeping():
    threading.Thread(target=_run_housekeeping, daemon=True).start()

@app.middleware("http")
async def _no_cache_mw(request: Request, call_next):
    resp: Response = await call_next(request)
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

@app.post("/api/analyze")
async def analyze_lesion(
    file: UploadFile = File(...), 
    payload: str = Form(...),
    stream: int = 0  # Add streaming support
):
    """Analyze lesion with enhanced LLM integration and streaming support."""
    logger.info(f"Analyze request: {file.filename} ({file.content_type}), stream={stream}")
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        payload_data = json.loads(payload)
        logger.info(f"Payload data: {payload_data}")
        lesion_request = LesionRequest(**payload_data)
        logger.info(f"Validated lesion request: {lesion_request}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Payload processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    run_id = _generate_run_id()
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save original image atomically
    image_ext = ''.join(pathlib.Path(file.filename).suffixes) or '.jpg'
    original_path = run_dir / f"original{image_ext}"
    contents = await file.read()
    _write_atomic(original_path, contents)

    # Build analysis prompt
    site_label = FACIAL_SUBUNIT_LABELS.get(lesion_request.subunit, lesion_request.subunit)
    
    # Extract basic image features (placeholder - would use CV in real implementation)
    cues = {
        "size_estimate": "8-12mm",
        "color": "pigmented",
        "border": "irregular" if lesion_request.ill_defined_borders else "well-defined",
        "surface": "varied"
    }
    
    flags = []
    if lesion_request.recurrent:
        flags.append("recurrent_lesion")
    if lesion_request.prior_histology:
        flags.append("prior_histology_available")
    
    # Template the prompts
    from server.llm.prompts import SYSTEM_PROMPT, VISION_PROMPT_TEMPLATE
    
    system = SYSTEM_PROMPT
    user_prompt = VISION_PROMPT_TEMPLATE.replace("{{ site }}", site_label)
    user_prompt = user_prompt.replace("{{ clinical_context }}", json.dumps({
        "site": site_label,
        "cues": cues,
        "flags": flags,
        "age": lesion_request.age,
        "sex": lesion_request.sex
    }))

    request_id = f"analyze_{run_id}"
    
    try:
        if stream:
            # Streaming response
            def stream_analysis():
                try:
                    accumulated_content = ""
                    chunk_generator = llm_router.complete_with_fallback(
                        system=system,
                        user=user_prompt,
                        schema=AnalysisOutput,
                        max_output_tokens=SETTINGS.max_output_tokens,
                        stream=True,
                        image_data=contents,
                        image_mime_type=file.content_type
                    )
                    
                    for chunk in chunk_generator:
                        accumulated_content += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    
                    # Final validation and completion
                    try:
                        import orjson
                        parsed_json = orjson.loads(accumulated_content)
                        validated_data = AnalysisOutput.model_validate(parsed_json)
                        
                        # Save analysis result
                        analysis_path = run_dir / 'analysis.json'
                        _write_atomic(analysis_path, orjson.dumps(validated_data.model_dump(), option=orjson.OPT_INDENT_2))
                        
                        # Generate additional artifacts (mock for now)
                        artifacts = _generate_mock_artifacts(run_dir, validated_data, lesion_request)
                        
                        final_response = {
                            "ok": True,
                            "run_id": run_id,
                            "analysis": validated_data.model_dump(),
                            "artifacts": artifacts
                        }
                        
                        yield f"data: {json.dumps({'type': 'complete', 'result': final_response})}\n\n"
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'error': f'Validation failed: {e}'})}\n\n"
                        
                except Exception as e:
                    logger.error(f"Streaming analysis failed: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            
            return StreamingResponse(stream_analysis(), media_type="text/event-stream")
        
        else:
            # Non-streaming response
            llm_response = llm_router.complete_with_fallback(
                system=system,
                user=user_prompt,
                schema=AnalysisOutput,
                max_output_tokens=SETTINGS.max_output_tokens,
                stream=False,
                image_data=contents,
                image_mime_type=file.content_type
            )
            
            # Log usage
            log_usage(
                llm_response.usage,
                request_id=request_id,
                endpoint="/api/analyze",
                additional_meta={"site": site_label, "stream": False}
            )
            
            analysis_result = llm_response.content
            
            # Save analysis result
            analysis_path = run_dir / 'analysis.json'
            import orjson
            _write_atomic(analysis_path, orjson.dumps(analysis_result, option=orjson.OPT_INDENT_2))
            
            # Generate additional artifacts
            artifacts = _generate_mock_artifacts(run_dir, AnalysisOutput(**analysis_result), lesion_request)
            
            return JSONResponse({
                "ok": True,
                "run_id": run_id,
                "analysis": analysis_result,
                "artifacts": artifacts,
                "provider": llm_response.usage.provider,
                "model": llm_response.usage.model
            })
            
    except LLMValidationError as e:
        logger.error(f"LLM validation error for {run_id}: {e}")
        raise HTTPException(status_code=422, detail=f"AI analysis validation failed: {str(e)[:500]}")
    except LLMError as e:
        logger.error(f"LLM error for {run_id}: {e}")
        raise HTTPException(status_code=503, detail=f"AI service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Analysis failed for {run_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def _generate_mock_artifacts(run_dir: pathlib.Path, analysis: AnalysisOutput, request: LesionRequest) -> dict:
    """Generate mock artifacts for demo purposes"""
    # In a real implementation, this would generate actual heatmaps, overlays, etc.
    
    # Create mock heatmap and overlay
    heatmap_path = run_dir / 'heatmap.png'
    overlay_path = run_dir / 'overlay.png'
    
    if not heatmap_path.exists():
        _write_atomic(heatmap_path, b'PNG heatmap placeholder')
    if not overlay_path.exists():
        _write_atomic(overlay_path, b'PNG overlay placeholder')
    
    # Create metrics
    metrics = {
        'run_id': str(run_dir.name),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'analysis': analysis.model_dump(),
        'input': request.model_dump()
    }
    metrics_path = run_dir / 'metrics.json'
    _write_atomic(metrics_path, json.dumps(metrics, indent=2).encode())
    
    # Generate report
    try:
        from . import report as report_builder
        report_builder.build_enhanced_report(
            run_id=str(run_dir.name),
            run_dir=run_dir,
            analysis=analysis,
            original_path=run_dir / "original.jpg",
            heatmap_path=heatmap_path,
            overlay_path=overlay_path,
            metrics=metrics
        )
    except Exception as e:
        logger.warning(f"Enhanced report generation failed: {e}")
        # Fallback to basic report
        try:
            from . import report as report_builder
            report_builder._fallback_pdf(
                run_dir / 'report.pdf', 
                {
                    "run_id": str(run_dir.name),
                    "created_at": metrics['created_at'],
                    "analysis": analysis.model_dump()
                }
            )
        except Exception:
            _write_atomic(run_dir / 'report.pdf', b'PDF generation failed')
    
    artifacts = {
        'original': f"/api/artifact/{run_dir.name}/original.jpg",
        'heatmap': f"/api/artifact/{run_dir.name}/{heatmap_path.name}",
        'overlay': f"/api/artifact/{run_dir.name}/{overlay_path.name}",
        'metrics': f"/api/artifact/{run_dir.name}/{metrics_path.name}",
        'analysis': f"/api/artifact/{run_dir.name}/analysis.json",
        'pdf': f"/api/report/{run_dir.name}.pdf",
    }
    
    return artifacts

@app.get("/api/artifact/{run_id}/{filename}")
async def get_artifact(run_id: str, filename: str):
    artifact_path = RUNS_ROOT / run_id / filename
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    etag = _hash_file(artifact_path)
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'ETag': etag
    }
    return FileResponse(artifact_path, headers=headers)

@app.get("/api/report/{run_id}.pdf")
async def get_report_pdf(run_id: str, regen: int | None = None):
    run_dir = RUNS_ROOT / run_id
    pdf_path = run_dir / 'report.pdf'
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    if regen:
        # attempt regeneration
        try:
            from . import report as report_builder
            metrics_path = run_dir / 'metrics.json'
            data = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            report_builder.build_report(
                run_id=run_id,
                run_dir=run_dir,
                lesion_probs=data.get('lesion_probs', {}),
                gate=data.get('gate', {}),
                original_path=next(run_dir.glob('original*')),
                heatmap_path=run_dir / 'heatmap.png',
                overlay_path=run_dir / 'overlay.png',
                metrics=data,
            )
        except Exception as e:
            logger.warning(f"Report regen failed: {e}")
    etag = _hash_file(pdf_path)
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'ETag': etag,
        'Content-Disposition': f'attachment; filename="SurgicalAI_{run_id}.pdf"'
    }
    return FileResponse(pdf_path, media_type='application/pdf', headers=headers)

@app.get("/api/run/{run_id}")
async def run_manifest(run_id: str):
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    files = []
    for p in run_dir.iterdir():
        if p.is_file():
            files.append({'name': p.name, 'size': p.stat().st_size})
    return {'run_id': run_id, 'files': files}

# Serve the static client from ./client
CLIENT_DIR = ROOT / "client"
if CLIENT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")

# Notes (for maintainers):
# • Enhanced with dual LLM provider support (OpenAI + Anthropic)
# • Structured outputs with Pydantic validation
# • Streaming support for real-time analysis
# • Cost and usage logging
# • Provider fallback and retry logic
