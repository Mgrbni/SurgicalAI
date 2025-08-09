from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from surgicalai.demo import run as demo_run

app = FastAPI(title="SurgicalAI")


@app.post("/demo")
async def demo_endpoint(with_llm: bool = False) -> dict:
    out = Path("api_demo")
    demo_run(out, with_llm=with_llm)
    return {"out": str(out)}


@app.get("/analyze")
async def analyze_endpoint():  # pragma: no cover - placeholder
    return {"status": "ok"}


@app.get("/plan")
async def plan_endpoint():  # pragma: no cover
    return {"status": "ok"}


@app.get("/validate")
async def validate_endpoint():  # pragma: no cover
    return {"status": "ok"}


@app.get("/visualize")
async def visualize_endpoint():  # pragma: no cover
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("surgicalai.api:app", host="0.0.0.0", port=8000)
