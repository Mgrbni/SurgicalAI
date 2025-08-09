from pydantic import BaseModel
import os, yaml, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
with open(ROOT / "settings.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

class AppSettings(BaseModel):
    model: str = os.getenv("OPENAI_MODEL", CFG.get("model", "gpt-5-mini"))
    temperature: float = CFG.get("temperature", 0.2)
    max_output_tokens: int = CFG.get("max_output_tokens", 800)
    caps: dict = CFG.get("caps", {})
    timeout_s: int = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))

SETTINGS = AppSettings()
