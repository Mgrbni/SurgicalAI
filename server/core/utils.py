"""Core utilities for SurgicalAI server."""

import os
import uuid
import logging
import base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple, Optional
from PIL import Image, UnidentifiedImageError
import io

from fastapi import HTTPException


def setup_logging():
    """Setup consistent logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def get_version_info() -> str:
    """Get version information from git or environment."""
    version = os.getenv("VERSION", "0.1.0")
    git_sha = os.getenv("GIT_SHA", "unknown")
    if git_sha != "unknown":
        return f"{version}+{git_sha[:8]}"
    return version


def generate_request_id() -> str:
    """Generate a unique request ID."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def validate_image(image_data: bytes, filename: Optional[str] = None) -> Tuple[Image.Image, dict]:
    """
    Validate and process uploaded image.
    
    Returns:
        Tuple of (PIL Image, metadata dict)
    
    Raises:
        HTTPException: If image is invalid or too large
    """
    # Check size limit (6MB as specified in requirements)
    if len(image_data) > 6 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image too large (>6MB). Try smaller or WebP format."
        )
    
    if len(image_data) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty image file"
        )
    
    try:
        # Open and validate image
        image = Image.open(io.BytesIO(image_data))
        image.verify()  # Verify it's a valid image
        
        # Reopen for actual use (verify() closes the file)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "filename": filename
        }
        
        return image, metadata
        
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported: JPEG, PNG, WebP"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image processing failed: {str(e)}"
        )


def get_artifact_path(path: str) -> Path:
    """
    Resolve artifact path securely, preventing directory traversal.
    
    Args:
        path: Relative path to artifact (e.g., "runs/123/overlay.png")
    
    Returns:
        Absolute path to artifact
    
    Raises:
        HTTPException: If path is invalid or outside allowed directory
    """
    # Root directory for artifacts
    root = Path(__file__).resolve().parents[2] / "runs"
    
    # Clean the path - remove any ../ attempts
    clean_path = Path(path).as_posix()
    if ".." in clean_path or clean_path.startswith("/"):
        raise HTTPException(status_code=403, detail="Invalid path")
    
    # Resolve full path
    full_path = root / clean_path
    
    # Security check - ensure it's within the allowed directory
    try:
        full_path.resolve().relative_to(root.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return full_path


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))
