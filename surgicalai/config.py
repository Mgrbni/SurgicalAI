from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    strain_thresh: float = 0.25
    vessel_risk_mm: List[int] = [5, 10]
    render_views: int = 8
    img_size: int = 512
    outputs_dir: str = "./outputs"


def load_config(path: Path | str = Path(__file__).with_name("config.yaml")) -> AppConfig:
    """Load application configuration from a YAML file."""
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    return AppConfig(**data)


config = load_config()
