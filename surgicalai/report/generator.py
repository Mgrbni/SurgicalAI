"""Report generation."""
from __future__ import annotations
from pathlib import Path
from ..utils.io import ensure_dir

def generate_report(run_id: str, content: dict, out_dir: Path):
    ensure_dir(out_dir)
    html = f"<html><body><h1>Report {run_id}</h1><pre>{content}</pre><p>Research prototype. Not for clinical use.</p></body></html>"
    with open(out_dir / 'report.html','w') as f:
        f.write(html)
    return out_dir / 'report.html'
