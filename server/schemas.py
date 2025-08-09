from pydantic import BaseModel, Field, conlist
from typing import List, Literal, Optional

class LesionBox(BaseModel):
    x: float
    y: float
    z: float
    radius_mm: float

class FlapPlan(BaseModel):
    technique: Literal["rotation","advancement","transposition"]
    vector: conlist(float, min_length=3, max_length=3) = Field(..., description="dx,dy,dz")
    along_langers: bool
    tension_zone: str
    predicted_success_pct: float

class LesionReport(BaseModel):
    diagnosis: Literal["melanoma","nevus","seb_keratosis","bcc","scc","other"]
    probability: float
    cam_hotspots: List[LesionBox]
    recommended_flap: Optional[FlapPlan]
    warnings: List[str] = []
