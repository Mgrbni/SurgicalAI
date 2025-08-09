from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

from surgicalai.config import LLMConfig
from surgicalai.logging_utils import get_logger
from surgicalai.utils.io import read_json

LOGGER = get_logger(__name__)

try:  # optional dependency
    import openai
except Exception:  # pragma: no cover
    openai = None


def redact_phi(data: Dict[str, Any]) -> Dict[str, Any]:
    return data


class OpenAIClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and openai is not None:
            openai.api_key = self.api_key

    def chat_json(self, system: str, user: str, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key or openai is None:
            LOGGER.warning("LLM disabled; returning stub message")
            return {k: f"stub {k}" for k in schema}
        for attempt in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model=self.config.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                content = response["choices"][0]["message"]["content"]
                return schema  # placeholder; real parsing skipped
            except Exception as exc:  # pragma: no cover
                LOGGER.error("LLM error: %s", exc)
                time.sleep(2 ** attempt)
        return {k: f"error {k}" for k in schema}

    def generate_narrative(self, case_dir: Path) -> Dict[str, Any]:
        lesion = read_json(case_dir / "lesion_probs.json")
        contraind = read_json(case_dir / "contraindications.json")
        plan = read_json(case_dir / "flap_plan.json")
        data = {"lesion": lesion, "contraindications": contraind, "plan": plan}
        if self.config.redact_phi:
            data = redact_phi(data)
        user = str(data)
        schema = {
            "summary": "",
            "risk_explanation": "",
            "flap_rationale": "",
            "alternatives": [],
            "disclaimer": "",
        }
        result = self.chat_json("surgical narrative", user, "narrative", schema)
        return result
