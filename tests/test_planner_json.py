from __future__ import annotations

from pathlib import Path

from surgicalai.demo import run as demo_run
from surgicalai.schemas import FlapPlan


def test_planner_json(tmp_path: Path) -> None:
    out = tmp_path / "case"
    demo_run(out)
    plan = FlapPlan.from_json(out / "flap_plan.json")
    assert plan.type == "rotation"
    assert len(plan.arc) > 0
