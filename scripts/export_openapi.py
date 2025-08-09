import json, os, pathlib
from importlib import import_module
# Import FastAPI app at src/surgicalai/api/main.py with variable "app"
mod = import_module("src.surgicalai.api.main")
app = getattr(mod, "app")
openapi = app.openapi()
out = pathlib.Path("docs/openapi.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(openapi, indent=2), encoding="utf-8")
print(f"Wrote {out}")
