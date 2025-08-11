"""Configuration management for SurgicalAI."""

from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
import os

# Feature flags
DISABLE_3D = True

def load_settings(settings_path: Path | str = "settings.yaml") -> Dict[str, Any]:
    """Load settings from YAML file with environment variable overrides.
    
    Args:
        settings_path: Path to settings YAML file
        
    Returns:
        Dict containing merged settings
    """
    # Load .env file
    load_dotenv()
    
    # Load base settings
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    
    # Override with environment variables
    env_prefix = "SURGICALAI_"
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            # Convert SURGICALAI_MODEL_NAME to ["model"]["name"]
            parts = key[len(env_prefix):].lower().split("_")
            
            # Walk the settings tree
            current = settings
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the final value
            current[parts[-1]] = value
            
    return settings
