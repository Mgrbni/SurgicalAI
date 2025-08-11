from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum

class SexEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"

class FacialSubunitEnum(str, Enum):
    FOREHEAD = "forehead"
    TEMPLE = "temple"
    EYEBROW = "eyebrow"
    EYELID_UPPER = "eyelid_upper"
    EYELID_LOWER = "eyelid_lower"
    CANTHUS_MEDIAL = "canthus_medial"
    CANTHUS_LATERAL = "canthus_lateral"
    NOSE_DORSUM = "nose_dorsum"
    NOSE_TIP = "nose_tip"
    NOSE_ALA = "nose_ala"
    CHEEK_MEDIAL = "cheek_medial"
    CHEEK_LATERAL = "cheek_lateral"
    ZYGOMATIC = "zygomatic"
    LIP_UPPER = "lip_upper"
    LIP_LOWER = "lip_lower"
    PHILTRUM = "philtrum"
    CHIN = "chin"
    JAWLINE = "jawline"
    EAR_HELIX = "ear_helix"
    EAR_LOBE = "ear_lobe"
    PAROTID_REGION = "parotid_region"
    NECK_ANTERIOR = "neck_anterior"

class LesionRequest(BaseModel):
    age: Optional[int] = None
    sex: SexEnum
    subunit: FacialSubunitEnum
    prior_histology: bool = False
    ill_defined_borders: bool = False
    recurrent: bool = False

class LesionBox(BaseModel):
    x: float
    y: float
    z: float
    radius_mm: float

class FlapPlan(BaseModel):
    technique: Literal["rotation","advancement","transposition"]
    vector: List[float] = Field(..., description="dx,dy,dz", min_length=3, max_length=3)
    along_langers: bool
    tension_zone: str
    predicted_success_pct: float

class LesionReport(BaseModel):
    diagnosis: Literal["melanoma","nevus","seb_keratosis","bcc","scc","other"]
    probability: float
    cam_hotspots: List[LesionBox]
    recommended_flap: Optional[FlapPlan]
    warnings: List[str] = []
