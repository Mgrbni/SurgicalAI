# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Type, get_origin

from pydantic import BaseModel

from surgicalai.config import LLMConfig
from surgicalai.logging_utils import get_logger
from surgicalai.utils.io import read_json, write_json

LOGGER = get_logger(__name__)

try:  # optional dependency
    import openai
except Exception:  # pragma: no cover
    openai = None


PHI_KEYS = {"name", "date", "mrn", "gps_exif", "notes"}


def redact_phi(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively remove PHI fields from a JSON-like structure."""

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k not in PHI_KEYS}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    return _clean(data)


class OpenAIClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and openai is not None:
            openai.api_key = self.api_key

    # ---- internal helpers -------------------------------------------------
    def _fallback(self, schema: Type[BaseModel]) -> BaseModel:
        fields: Dict[str, Any] = {}
        for name, field in schema.__fields__.items():  # type: ignore[attr-defined]
            origin = get_origin(field.outer_type_) or field.outer_type_
            fields[name] = [] if origin is list else "LLM response unavailable"
        return schema(**fields)

    def _estimate_cost(self, usage: Dict[str, int]) -> float:
        pricing = {"gpt-4o": (0.005, 0.015)}  # per 1K tokens (prompt, completion)
        prompt_cost, completion_cost = pricing.get(self.config.model, (0.0, 0.0))
        return (
            usage.get("prompt_tokens", 0) / 1000 * prompt_cost
            + usage.get("completion_tokens", 0) / 1000 * completion_cost
        )

    def _log_usage(self, case_dir: Path, usage: Dict[str, int]) -> None:
        record = {
            "model": self.config.model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cost": round(self._estimate_cost(usage), 6),
        }
        path = case_dir / "llm_usage.json"
        if path.exists():
            data = read_json(path)
            if isinstance(data, list):
                data.append(record)
            else:
                data = [data, record]
        else:
            data = [record]
        write_json(path, data)  # type: ignore[arg-type]

    # ---- API interaction ---------------------------------------------------
    def chat_json(
        self, system: str, user: str, schema: Type[BaseModel]
    ) -> Tuple[BaseModel, Dict[str, int] | None]:
        if not self.api_key or openai is None:
            LOGGER.warning("LLM disabled; returning stub message")
            return self._fallback(schema), None
        for attempt in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response["choices"][0]["message"]["content"]
                usage = response.get("usage", {})
                try:
                    data = json.loads(content)
                    result = schema.parse_obj(data)
                except Exception as exc:
                    LOGGER.warning("Invalid LLM JSON: %s", exc)
                    result = self._fallback(schema)
                return result, usage
            except Exception as exc:  # pragma: no cover - transient API errors
                LOGGER.error("LLM error: %s", exc)
                time.sleep(2**attempt)
        return self._fallback(schema), None

    # ---- public API --------------------------------------------------------
    def generate_narrative(self, case_dir: Path) -> Dict[str, Any]:
        lesion = read_json(case_dir / "lesion_probs.json")
        contraind = read_json(case_dir / "contraindications.json")
        plan = read_json(case_dir / "flap_plan.json")
        data = {"lesion": lesion, "contraindications": contraind, "plan": plan}
        if self.config.redact_phi:
            data = redact_phi(data)
        user = json.dumps(data)

        from surgicalai.schemas import Narrative

        result, usage = self.chat_json("surgical narrative", user, Narrative)
        if usage:
            self._log_usage(case_dir, usage)
        return result.dict()
