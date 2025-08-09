from __future__ import annotations

import os
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


class TrainConfig(BaseModel):
    img_size: int = 256
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 5
    class_names: list[str] = ["benign", "melanoma"]


class DataConfig(BaseModel):
    source: str = "folder"
    root: str = "data/lesions"
    csv: str = "data/lesions/labels.csv"
    db_dsn: str = "postgresql+psycopg2://user:pass@host/dbname"


class ExportConfig(BaseModel):
    to_onnx: bool = True
    to_torchscript: bool = True


class Config(BaseModel):
    analyze: AnalyzeConfig = AnalyzeConfig()
    plan: PlanConfig = PlanConfig()
    validate: ValidateConfig = ValidateConfig()
    llm: LLMConfig = LLMConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    export: ExportConfig = ExportConfig()


def load_config(path: Path | None = None) -> Config:
    path = path or Path(__file__).with_name("config.yaml")
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        cfg = Config(**data)
    else:
        cfg = Config()

    # environment overrides
    if os.getenv("LLM_ENABLED") is not None:
        cfg.llm.enabled = os.getenv("LLM_ENABLED", "").lower() == "true"
    model = os.getenv("OPENAI_MODEL")
    if model:
        cfg.llm.model = model
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        cfg.data.db_dsn = db_url
    return cfg
