from pydantic import BaseModel, Field
import os, yaml, pathlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cost table for token pricing (USD per 1K tokens)
COST_TABLE = {
    "openai": {
        "gpt-4o": {"input": 0.0025, "output": 0.010},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }
}

def get_cost_estimate(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost estimate for token usage"""
    try:
        rates = COST_TABLE.get(provider, {}).get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        return round(input_cost + output_cost, 6)
    except Exception:
        return 0.0

def redact_secret(value: str) -> str:
    """Redact API keys and secrets for logging"""
    if not value or len(value) < 8:
        return "***"
    return f"{value[:8]}***{value[-4:]}"

ROOT = pathlib.Path(__file__).resolve().parents[1]
with open(ROOT / "settings.yaml", "r") as f:
    CFG = yaml.safe_load(f) or {}

def expand_env_vars(value):
    """Expand environment variables in string values"""
    if isinstance(value, str) and "${" in value:
        # Simple env var expansion - can be improved
        import re
        def replacer(match):
            var_with_default = match.group(1)
            if ":-" in var_with_default:
                var, default = var_with_default.split(":-", 1)
                return os.getenv(var, default)
            else:
                return os.getenv(var_with_default, "")
        return re.sub(r'\$\{([^}]+)\}', replacer, value)
    return value

class AppSettings(BaseModel):
    # LLM Provider Configuration
    provider: str = expand_env_vars(CFG.get("provider", "openai"))
    openai_model: str = expand_env_vars(CFG.get("openai_model", "gpt-4o-mini"))
    anthropic_model: str = expand_env_vars(CFG.get("anthropic_model", "claude-3-5-sonnet-20241022"))
    max_output_tokens: int = int(expand_env_vars(str(CFG.get("max_output_tokens", 1200))))
    timeout_seconds: int = int(expand_env_vars(str(CFG.get("timeout_seconds", 45))))
    
    # API Keys (from environment)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_project: str = os.getenv("OPENAI_PROJECT", "")
    openai_org: str = os.getenv("OPENAI_ORG", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Legacy LLM settings (backwards compatibility)
    model: str = expand_env_vars(CFG.get("model", "gpt-4o-mini"))
    caps: dict = CFG.get("caps", {})
    timeout_s: int = int(expand_env_vars(str(CFG.get("timeout_seconds", 45))))

    # Vision classifier settings (demo; NOT FOR CLINICAL USE)
    class_names: list[str] = CFG.get("class_names", [
        "melanoma","bcc","scc","nevus","seborrheic_keratosis","benign_other"
    ])
    temperature: float = float(CFG.get("temperature", 1.5))
    class_weight: dict[str, float] = CFG.get("class_weight", {c: 1.0 for c in [
        "melanoma","bcc","scc","nevus","seborrheic_keratosis","benign_other"
    ]})
    class_bias: dict[str, float] = CFG.get("class_bias", {c: 0.0 for c in [
        "melanoma","bcc","scc","nevus","seborrheic_keratosis","benign_other"
    ]})
    min_conf_for_gate: float = float(CFG.get("min_conf_for_gate", 0.55))
    melanoma_defer_threshold: float = float(CFG.get("melanoma_defer_threshold", 0.30))

    # Modes
    photo_mode: bool = bool(CFG.get("photo_mode", True))  # legacy alias for 2D only
    polycam_mode: bool = bool(CFG.get("polycam_mode", False))  # enable 3D mesh ingestion paths
    
    # Heatmap & ROI settings
    heatmap: dict = CFG.get("heatmap", {})
    
    # Vision-LLM Fusion settings
    fusion: dict = CFG.get("fusion", {})

SETTINGS = AppSettings()
