# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from surgicalai.utils.io import read_json, write_json


class LesionProbs(BaseModel):
    class_probs: Dict[str, float]
    top_class: str

    def to_json(self, path: Path) -> None:
        write_json(path, self.dict())

    @classmethod
    def from_json(cls, path: Path) -> "LesionProbs":
        return cls(**read_json(path))


class FlapPlan(BaseModel):
    type: str
    pivot: List[float]
    arc: List[List[float]]
    tension_axis: List[float]
    success_prob: float
    notes: str = ""

    def to_json(self, path: Path) -> None:
        write_json(path, self.dict())

    @classmethod
    def from_json(cls, path: Path) -> "FlapPlan":
        return cls(**read_json(path))


class Alert(BaseModel):
    type: str
    distance_mm: float
    details: str


class Contraindications(BaseModel):
    alerts: List[Alert]
    score: float

    def to_json(self, path: Path) -> None:
        write_json(path, self.dict())

    @classmethod
    def from_json(cls, path: Path) -> "Contraindications":
        return cls(**read_json(path))


class Narrative(BaseModel):
    summary: str
    risk_explanation: str
    flap_rationale: str
    alternatives: List[Dict[str, str]]
    disclaimer: str

    def to_json(self, path: Path) -> None:
        write_json(path, self.dict())

    @classmethod
    def from_json(cls, path: Path) -> "Narrative":
        return cls(**read_json(path))
