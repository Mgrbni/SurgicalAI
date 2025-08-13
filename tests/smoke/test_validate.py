import json
import subprocess
import sys
from pathlib import Path


def run_cli(args):
    cmd = [sys.executable, "-m", "SurgicalAI.cli"] + args
    out = subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[2])
    return out.decode("utf-8").strip().splitlines()[-1]


def test_validate_over_golden(tmp_path):
    # ensure golden exists
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_golden", str(Path("tests/golden/make_golden.py")))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    out_dir = str(tmp_path / "out")
    line = run_cli(["validate", "tests/golden", "--out", out_dir])
    payload = json.loads(line)
    assert payload["ok"] is True
    assert payload["total"] == 10
