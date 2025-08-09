from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_exists(path: Union[str, Path]) -> Path:
    """Validate that the path exists."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return p
