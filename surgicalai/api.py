# SPDX-License-Identifier: Apache-2.0
"""Minimal FastAPI app for SurgicalAI."""

from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI

app = FastAPI(title="SurgicalAI")


@app.get("/ping")
def ping() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.post("/predict")
def predict() -> dict[str, float]:
    """Return stub prediction on synthetic data."""
    return {"lesion_prob": 0.5}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("surgicalai.api:app", host="0.0.0.0", port=8000)
