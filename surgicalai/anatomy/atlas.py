"""Load anatomy atlas data."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from ..utils.io import load_json
from ..config import CONFIG


def load_atlas() -> Dict[str, Any]:
    base = CONFIG.data / 'anatomy'
    langer = load_json(base / 'langer_lines.json')
    vessels = load_json(base / 'vessels.json')
    schema = load_json(base / 'landmarks_schema.json')
    return {'langer': langer, 'vessels': vessels, 'schema': schema}
