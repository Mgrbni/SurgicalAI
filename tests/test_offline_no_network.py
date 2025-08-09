from __future__ import annotations

# SPDX-License-Identifier: Apache-2.0

import requests
from pathlib import Path
from surgicalai.demo import run


def test_no_external_calls_when_offline(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def fake_post(*args, **kwargs):  # type: ignore[unused-argument]
        calls.append(1)
        raise AssertionError("network used")

    monkeypatch.setattr(requests, "post", fake_post)
    run(tmp_path, with_llm=False)
    assert not calls
