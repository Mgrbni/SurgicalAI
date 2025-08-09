# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))


def load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.savez(path, **arrays)
