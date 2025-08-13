"""SurgicalAI FastAPI Application - Clean Architecture

Unified server application with proper routing and static file serving.
Implements the requirements: zero 404s, single analyze endpoint, structured response.
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .core.utils import setup_logging, get_version_info


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).resolve().parents[1]
CLIENT_DIR = ROOT / "client"
RUNS_DIR = ROOT / "runs"

# Ensure runs directory exists
RUNS_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("SurgicalAI server starting...")
    logger.info(f"Version: {get_version_info()}")
    logger.info(f"Client dir: {CLIENT_DIR}")
    logger.info(f"Runs dir: {RUNS_DIR}")
    yield
    logger.info("SurgicalAI server shutting down...")


app = FastAPI(
    title="SurgicalAI",
    description="AI-powered surgical planning for dermatologic reconstruction",
    version=get_version_info(),
    lifespan=lifespan
)

# CORS configuration
cors_origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative dev server
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# Add same-origin for production
if os.getenv("PRODUCTION"):
    cors_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Health endpoints at root level
@app.get("/healthz")
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": get_version_info()}

@app.get("/version")
def version_info():
    """Version information endpoint."""
    return {"version": get_version_info(), "git_sha": os.getenv("GIT_SHA", "unknown")}

# Serve SPA under /app and redirect root/legacy to it
if CLIENT_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")
    logger.info(f"Serving client files from {CLIENT_DIR} at /app")
else:
    logger.warning(f"Client directory not found: {CLIENT_DIR}")

@app.get("/")
def root_to_app():
    """Redirect root to /app/ for unified UI."""
    return RedirectResponse(url="/app/", status_code=307)
