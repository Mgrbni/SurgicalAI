from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel
import yaml


class AnalyzeConfig(BaseModel):
    gradcam_threshold: float = 0.6
    n_views: int = 8


class PlanConfig(BaseModel):
    arc_multiplier: float = 3.5
    tension_angle_deg: list[float] = [30, 45]


class ValidateConfig(BaseModel):
    danger_margin_mm: float = 5.0


class LLMConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o"
    max_tokens: int = 700
    temperature: float = 0.2
    timeout_s: int = 30
    redact_phi: bool = True


class Config(BaseModel):
    analyze: AnalyzeConfig = AnalyzeConfig()
    plan: PlanConfig = PlanConfig()
    validate: ValidateConfig = ValidateConfig()
    llm: LLMConfig = LLMConfig()


def load_config(path: Path | None = None) -> Config:
    path = path or Path(__file__).with_name("config.yaml")
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        return Config(**data)
    return Config()
