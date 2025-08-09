"""Configuration utilities."""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = {
    'paths': {
        'data': 'data',
        'models': 'models',
        'outputs': 'outputs'
    },
    'render_views': 8,
    'img_size': 512,
    'strain_thresh': 0.25,
    'vessel_risk_mm': [5,10]
}

CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'

@dataclass
class Config:
    data: Path = Path('data')
    models: Path = Path('models')
    outputs: Path = Path('outputs')
    render_views: int = 8
    img_size: int = 512
    strain_thresh: float = 0.25
    vessel_risk_mm: tuple[int,int] = (5,10)

    @classmethod
    def load(cls) -> 'Config':
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(open(CONFIG_PATH))
        else:
            cfg = DEFAULT_CONFIG
        paths = cfg.get('paths', {})
        return cls(
            data=Path(paths.get('data', 'data')),
            models=Path(paths.get('models', 'models')),
            outputs=Path(paths.get('outputs', 'outputs')),
            render_views=cfg.get('render_views',8),
            img_size=cfg.get('img_size',512),
            strain_thresh=cfg.get('strain_thresh',0.25),
            vessel_risk_mm=tuple(cfg.get('vessel_risk_mm',[5,10]))
        )

CONFIG = Config.load()
