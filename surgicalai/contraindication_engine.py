"""Stub contraindication checker."""
from __future__ import annotations

from typing import Dict, Any


def evaluate(anatomy: Dict[str, Any]) -> Dict[str, Any]:
    """Return placeholder contraindication information."""
    return {"risk_flags": [], "notes": "no contraindications"}
