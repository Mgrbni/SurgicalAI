import os
import json
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from .settings import SETTINGS
from .schemas import LesionReport
from .rate_limit import retry_policy

# Validate key early
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="SurgicalAI API")

# CORS (be permissive for demo; lock down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static client from ./client
ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIENT_DIR = ROOT / "client"
if CLIENT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")

class InferenceReq(BaseModel):
    prompt: str
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    json_schema: bool = False  # set true for lesion/flap JSON

@app.get("/healthz")
def healthz():
    return {"ok": True, "model": SETTINGS.model}

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
    temp = SETTINGS.temperature if req.temperature is None else req.temperature
    max_tokens = min(req.max_output_tokens or SETTINGS.max_output_tokens,
                     SETTINGS.caps.get("max_tokens_per_request", 2000))

    def call():
        if req.json_schema:
            # Structured Outputs: enforce Pydantic schema
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "lesion_report",
                    "strict": True,
                    "schema": LesionReport.model_json_schema(),
                },
            }
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": req.prompt}],
                temperature=temp,
                max_tokens=max_tokens,
                timeout=SETTINGS.timeout_s,
                response_format=response_format,
            )
            content = resp.choices[0].message.content
            parsed = LesionReport.model_validate_json(content)
            data = parsed.model_dump()
            return JSONResponse({"ok": True, "data": data})
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": req.prompt}],
                temperature=temp,
                max_tokens=max_tokens,
                timeout=SETTINGS.timeout_s,
            )
            text = resp.choices[0].message.content
            return {"ok": True, "text": text}

    try:
        return retry_policy(call)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stream")  # streaming for long replies
def stream(req: InferenceReq):
    model = req.model or SETTINGS.model
    temp = SETTINGS.temperature if req.temperature is None else req.temperature
    max_tokens = min(req.max_output_tokens or SETTINGS.max_output_tokens,
                     SETTINGS.caps.get("max_tokens_per_request", 2000))

    def gen():
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": req.prompt}],
                temperature=temp,
                max_tokens=max_tokens,
                timeout=SETTINGS.timeout_s,
                stream=True,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n\n[STREAM_ERROR] {e}"

    return StreamingResponse(gen(), media_type="text/plain")

# Notes (for maintainers):
# • Adjusted to use OpenAI's chat.completions API for create and stream.
# • Structured outputs use json_schema response_format with Pydantic model's JSON schema.
# • Parsing and validation handled via Pydantic's model_validate_json.
# • Streaming does not support structured outputs.