import os
import pathlib

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from server.settings import SETTINGS
from server.schemas import LesionReport
from server.rate_limit import retry_policy

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Validate key early
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=OPENAI_API_KEY)

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

app = FastAPI(title="SurgicalAI API")

# CORS (be permissive for demo; lock down in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    # CREATE (non-JSON mode)
    def call_create(prompt: str, model: str, temp: float, n: int):
        base = dict(
            model=model,
            input=[{"role": "user", "content": prompt}],
            temperature=temp,
            timeout=SETTINGS.timeout_s,
        )
        try:
            return client.responses.create(**_with_max_output(base, n))
        except Exception as e:
            if _needs_token_fallback(e):
                return client.responses.create(**_with_max_completion(base, n))
            raise

    # PARSE (Structured Outputs / json_schema=True)
    def call_parse(prompt: str, model: str, temp: float, n: int):
        base = dict(
            model=model,
            input=prompt,
            temperature=temp,
            response_format=LesionReport,
            timeout=SETTINGS.timeout_s,
        )
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
    temp = SETTINGS.temperature if req.temperature is None else req.temperature
    max_tokens = min(req.max_output_tokens or SETTINGS.max_output_tokens,
                     SETTINGS.caps.get("max_tokens_per_request", 2000))

    def _stream_kwargs(prompt, model, temp, n):
        return dict(
            model=model,
            input=[{"role": "user", "content": prompt}],
            temperature=temp,
            timeout=SETTINGS.timeout_s,
        )

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
