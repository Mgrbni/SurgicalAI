# SPDX-License-Identifier: Apache-2.0
"""Utility helpers for SurgicalAI."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed Python and NumPy for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: Any, path: Path | str) -> None:
    """Write *data* to *path* as pretty JSON."""
    Path(path).write_text(json.dumps(data, indent=2))


def load_json(path: Path | str) -> Any:
    """Read JSON from *path* and return the parsed object."""
    return json.loads(Path(path).read_text())
