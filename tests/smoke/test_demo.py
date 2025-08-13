import json
import subprocess
import sys
from pathlib import Path


def run_cli(args):
    cmd = [sys.executable, "-m", "SurgicalAI.cli"] + args
    out = subprocess.check_output(cmd, cwd=Path(__file__).resolve().parents[2])
    return out.decode("utf-8").strip().splitlines()[-1]


def test_demo_runs_and_outputs(tmp_path):
    img = str(Path("assets/demo_face.jpg"))
    out_dir = str(tmp_path / "out")
    line = run_cli(["demo", img, "--out", out_dir])
    payload = json.loads(line)
    assert payload["ok"] is True
    assert payload["flap_candidates"]
    for p in payload["artifacts"]:
        assert Path(p).exists()
