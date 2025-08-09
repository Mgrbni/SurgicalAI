"""IO helpers."""
from pathlib import Path
from typing import Any
import json


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
