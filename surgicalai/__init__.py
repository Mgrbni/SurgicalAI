# SPDX-License-Identifier: Apache-2.0
"""SurgicalAI prototype package."""

from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0

import os

os.environ.setdefault("SURGICALAI_OFFLINE_LLM", "1")

try:  # load environment variables from .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

# force CPU/offscreen rendering for environments without a GUI
os.environ.setdefault("OPEN3D_CPU_RENDERING", "1")

__all__: list[str] = []
__version__ = "0.1.0"
