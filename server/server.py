import os
import pathlib
import json
import logging
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from server.settings import SETTINGS
from server.schemas import LesionReport, LesionRequest, FacialSubunitEnum
from server.rate_limit import retry_policy

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate key early
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=OPENAI_API_KEY)

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

# --- token fallback helpers ---
def _with_max_output(kwargs: dict, n: int):
    k = dict(kwargs)
    k.pop("max_completion_tokens", None)
    k["max_output_tokens"] = n
    return k


def _with_max_completion(kwargs: dict, n: int):
    k = dict(kwargs)
    k.pop("max_output_tokens", None)
    k["max_completion_tokens"] = n
    return k


def _needs_token_fallback(exc: Exception) -> bool:
    s = str(exc).lower()
    return "unsupported parameter" in s and ("max_tokens" in s or "max_output_tokens" in s)

NO_TEMP_MODELS = (
    "o1",
    "o1-mini",
    "o3",
    "o3-mini",
    "gpt-5",
    "gpt-5-thinking",
)


def _supports_temperature(model: str) -> bool:
    return not any(model.startswith(m) for m in NO_TEMP_MODELS)

app = FastAPI(title="SurgicalAI API")

# CORS (be permissive for demo; lock down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:5173", "http://127.0.0.1:8000", "http://127.0.0.1:5173"],
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
    return {"ok": True, "model": SETTINGS.model}

@app.get("/api/facial-subunits")
def get_facial_subunits():
    """Return curated facial subunit options."""
    subunits = []
    for value, label in FACIAL_SUBUNIT_LABELS.items():
        subunits.append({"value": value, "label": label})
    return {"subunits": subunits}

@app.post("/api/analyze")
async def analyze_lesion(
    file: UploadFile = File(...),
    payload: str = Form(...)
):
    """Analyze lesion with multipart/form-data (file + JSON payload)."""
    logger.info(f"Received analyze request - file: {file.filename}, content_type: {file.content_type}")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Parse JSON payload
        payload_data = json.loads(payload)
        lesion_request = LesionRequest(**payload_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in payload")
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    
    try:
        # Save file to demo directory
        runs_dir = ROOT / "runs" / "demo"
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded image
        image_path = runs_dir / file.filename
        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Generate mock probabilities (in real implementation, this would call ML model)
        import random
        random.seed(42)  # For consistent demo results
        
        lesion_probs = {
            "melanoma": round(random.uniform(0.1, 0.8), 2),
            "nevus": round(random.uniform(0.1, 0.4), 2),
            "seb_keratosis": round(random.uniform(0.05, 0.3), 2),
            "bcc": round(random.uniform(0.05, 0.2), 2),
            "scc": round(random.uniform(0.02, 0.15), 2)
        }
        
        # Normalize probabilities
        total = sum(lesion_probs.values())
        lesion_probs = {k: round(v/total, 2) for k, v in lesion_probs.items()}
        
        # Generate gate decision based on melanoma probability
        melanoma_prob = lesion_probs.get("melanoma", 0)
        allow_flap = melanoma_prob < 0.3
        gate_reason = "Low melanoma suspicion" if allow_flap else "High melanoma suspicion"
        gate_guidance = "Consider flap reconstruction" if allow_flap else "Biopsy/WLE first"
        
        # Create mock artifacts
        overlay_path = runs_dir / "overlay.png"
        heatmap_path = runs_dir / "heatmap.png" 
        report_path = runs_dir / "report.pdf"
        
        # Create placeholder files if they don't exist
        if not overlay_path.exists():
            overlay_path.write_bytes(b"PNG placeholder")
        if not heatmap_path.exists():
            heatmap_path.write_bytes(b"PNG placeholder")
        if not report_path.exists():
            report_path.write_bytes(b"PDF placeholder")
        
        result = {
            "ok": True,
            "lesion_probs": lesion_probs,
            "gate": {
                "allow_flap": allow_flap,
                "reason": gate_reason,
                "guidance": gate_guidance
            },
            "artifacts": {
                "overlay_png": f"/api/artifact/demo/overlay.png",
                "heatmap_png": f"/api/artifact/demo/heatmap.png", 
                "report_pdf": f"/api/artifact/demo/report.pdf"
            }
        }
        
        logger.info(f"Analysis completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/artifact/{path:path}")
async def get_artifact(path: str):
    """Serve artifact files from runs directory."""
    artifact_path = ROOT / "runs" / path
    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # Security check - ensure path is within runs directory
    try:
        artifact_path.resolve().relative_to((ROOT / "runs").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(artifact_path)

@app.api_route("/api/infer", methods=["GET", "POST"])  # non-stream simple text or structured JSON
def infer(
    req: InferenceReq | None = None,
    prompt: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    json_schema: bool = False,
):
    if req is None:
        if prompt is None:
            raise HTTPException(status_code=400, detail="prompt is required")
        req = InferenceReq(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
        )

    model = req.model or SETTINGS.model
    temp = req.temperature
    max_tokens = min(req.max_output_tokens or SETTINGS.max_output_tokens,
                     SETTINGS.caps.get("max_tokens_per_request", 2000))
    # CREATE (non-JSON mode)
    def call_create(prompt: str, model: str, temp: float | None, n: int):
        base = dict(
            model=model,
            input=[{"role": "user", "content": prompt}],
            timeout=SETTINGS.timeout_s,
        )
        if temp is not None and _supports_temperature(model):
            base["inference_config"] = {"temperature": temp}
        try:
            return client.responses.create(**_with_max_output(base, n))
        except Exception as e:
            if _needs_token_fallback(e):
                return client.responses.create(**_with_max_completion(base, n))
            raise

    # PARSE (Structured Outputs / json_schema=True)
    def call_parse(prompt: str, model: str, temp: float | None, n: int):
        base = dict(
            model=model,
            input=prompt,
            response_format=LesionReport,
            timeout=SETTINGS.timeout_s,
        )
        if temp is not None and _supports_temperature(model):
            base["inference_config"] = {"temperature": temp}
        try:
            return client.responses.parse(**_with_max_output(base, n))
        except Exception as e:
            if _needs_token_fallback(e):
                return client.responses.parse(**_with_max_completion(base, n))
            raise

    def call():
        if req.json_schema:
            resp = call_parse(req.prompt, model, temp, max_tokens)
            parsed = resp.output_parsed
            data = parsed.dict() if hasattr(parsed, "dict") else parsed
            return JSONResponse({"ok": True, "data": data})
        else:
            resp = call_create(req.prompt, model, temp, max_tokens)
            return {"ok": True, "text": resp.output_text}

    try:
        return retry_policy(call)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stream")  # streaming for long replies
def stream(req: InferenceReq):
    model = req.model or SETTINGS.model
    temp = req.temperature
    max_tokens = min(req.max_output_tokens or SETTINGS.max_output_tokens,
                     SETTINGS.caps.get("max_tokens_per_request", 2000))

    def _stream_kwargs(prompt, model, temp, n):
        k = dict(
            model=model,
            input=[{"role": "user", "content": prompt}],
            timeout=SETTINGS.timeout_s,
        )
        if temp is not None and _supports_temperature(model):
            k["inference_config"] = {"temperature": temp}
        return k

    def gen():
        base = _stream_kwargs(req.prompt, model, temp, max_tokens)
        try:
            ctx = client.responses.stream(**_with_max_output(base, max_tokens))
            use_completion = False
        except Exception as e:
            if _needs_token_fallback(e):
                ctx = client.responses.stream(**_with_max_completion(base, max_tokens))
                use_completion = True
            else:
                yield f"\n\n[STREAM_ERROR] {e}"
                return

        with ctx as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
            stream.close()

    return StreamingResponse(gen(), media_type="text/plain")

# Serve the static client from ./client
CLIENT_DIR = ROOT / "client"
if CLIENT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")

# Notes (for maintainers):
# • Uses OpenAI's Responses API for create, parse, and stream.
# • Automatically falls back between max_output_tokens and max_completion_tokens.
# • Structured outputs leverage responses.parse with the LesionReport Pydantic model.
# • Streaming does not support structured outputs.
