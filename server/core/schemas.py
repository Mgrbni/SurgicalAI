"""Pydantic schemas for SurgicalAI API."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class ROI(BaseModel):
    """Region of Interest definition."""
    x: float = Field(..., description="X coordinate (normalized 0-1)")
    y: float = Field(..., description="Y coordinate (normalized 0-1)")
    width: float = Field(..., description="Width (normalized 0-1)")
    height: float = Field(..., description="Height (normalized 0-1)")


class RiskFactors(BaseModel):
    """Patient risk factors and clinical flags."""
    age: Optional[int] = None
    sex: Optional[str] = None
    h_zone: bool = False
    ill_defined_borders: bool = False
    recurrent_tumor: bool = False
    prior_histology: bool = False


class AnalysisRequest(BaseModel):
    """Request for lesion analysis."""
    image_base64: Union[str, bytes] = Field(..., description="Base64 encoded image data")
    roi: Union[ROI, Dict[str, float]] = Field(..., description="Region of interest")
    site: Optional[str] = Field(None, description="Anatomical site/facial subunit")
    risk_factors: Optional[Union[RiskFactors, Dict[str, Any]]] = Field(default_factory=dict)
    offline: bool = Field(False, description="Skip LLM analysis, use heuristics only")
    provider: Optional[str] = Field(None, description="AI provider: auto|openai|anthropic")
    model: Optional[str] = Field(None, description="Model name per provider")


class DiagnosisProb(BaseModel):
    """Diagnosis probability entry."""
    label: str = Field(..., description="Diagnosis label")
    prob: float = Field(..., description="Probability score (0-1)")


class Diagnostics(BaseModel):
    """Diagnostic analysis results."""
    top3: List[DiagnosisProb] = Field(..., description="Top 3 diagnoses with probabilities")
    notes: str = Field("", description="Model rationale or notes")


class ABCDEAnalysis(BaseModel):
    """ABCDE rule analysis."""
    asymmetry: float = Field(0.0, description="Asymmetry score (0-1)")
    border: float = Field(0.0, description="Border irregularity score (0-1)")
    color: float = Field(0.0, description="Color variation score (0-1)")
    diameter: float = Field(0.0, description="Diameter score (0-1)")
    evolving: float = Field(0.0, description="Evolution score (0-1)")
    overall_score: float = Field(0.0, description="Overall ABCDE score (0-1)")


class FlapParameters(BaseModel):
    """Flap design parameters."""
    defect_d_mm: float = Field(..., description="Defect diameter in mm")
    margin_mm: float = Field(..., description="Surgical margin in mm")
    arc_deg: float = Field(..., description="Flap arc in degrees")
    base_width_mm: float = Field(..., description="Flap base width in mm")
    angle_to_rstl_deg: float = Field(..., description="Angle to RSTL in degrees")
    geometry: str = Field("", description="Flap geometry information")
    indications: str = Field("", description="Indications for flap use")
    contraindications: str = Field("", description="Contraindications for flap use")
    howto: str = Field("", description="How-to information for flap use")


class FlapAlternative(BaseModel):
    """Alternative flap option."""
    type: str = Field(..., description="Flap type")
    score: float = Field(..., description="Suitability score (0-1)")


class FlapPlan(BaseModel):
    """Surgical flap planning results."""
    type: str = Field(..., description="Primary flap type")
    params: FlapParameters = Field(..., description="Flap parameters")
    alternatives: List[FlapAlternative] = Field(default_factory=list)


class Artifacts(BaseModel):
    """Generated artifacts (images, overlays)."""
    overlay_png_base64: Optional[str] = Field(None, description="Overlay image as base64")
    heatmap_png_base64: Optional[str] = Field(None, description="Heatmap image as base64")
    thumbnails: List[str] = Field(default_factory=list, description="Thumbnail images as base64")


class AnalysisResponse(BaseModel):
    """Response from lesion analysis."""
    diagnostics: Diagnostics = Field(..., description="Diagnostic results")
    abcde: Optional[ABCDEAnalysis] = Field(None, description="ABCDE analysis")
    rstl_angle_deg: float = Field(..., description="RSTL angle in degrees")
    tension_score: float = Field(..., description="Tension score (0-1)")
    flap_plan: FlapPlan = Field(..., description="Surgical flap plan")
    artifacts: Artifacts = Field(..., description="Generated artifacts")
    citations: List[str] = Field(default_factory=list, description="Literature citations")
    request_id: str = Field(..., description="Unique request identifier")


class ReportRequest(BaseModel):
    """Request for PDF report generation."""
    analysis_payload: Dict[str, Any] = Field(..., description="Complete analysis results")
    doctor_name: Optional[str] = Field("Dr. Mehdi Ghorbani Karimabad", description="Doctor name for signature")
    patient_id: Optional[str] = Field(None, description="Patient identifier (anonymized)")
    clinic_name: Optional[str] = Field("SurgicalAI", description="Clinic name")


# === Minimal Clinical Demo (Triage) ===

class TriageRisks(BaseModel):
    smoker: bool = False
    uv_high: bool = False
    immunosuppressed: bool = False
    radiated: bool = False
    anticoagulants: bool = False
    diabetes: bool = False
    ill_defined_borders: bool = False
    recurrent_tumor: bool = False


class TriageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image data (data URL or raw base64)")
    roi_polygon: List[List[float]] = Field(..., description="ROI polygon points as [[x,y], ...] normalized 0-1")
    site: str = Field(..., description="Anatomical site/facial subunit")
    age: Optional[int] = Field(None, description="Patient age")
    sex: Optional[str] = Field(None, description="Patient sex (M/F)")
    risks: TriageRisks = Field(default_factory=TriageRisks)
    offline: bool = Field(True, description="Use offline heuristics only")


class TriageDiagnostics(BaseModel):
    top3: List[DiagnosisProb]
    notes: str


class TriageOncology(BaseModel):
    tumor_size_mm: float
    recommended_margin_mm: float
    technique: str
    gates: Dict[str, bool]


class TriageReconstruction(BaseModel):
    primary: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    angle_to_rstl_deg: float


class TriageSafety(BaseModel):
    h_zone: bool
    nerve_vessel_alerts: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class TriageOverlay(BaseModel):
    excision: Dict[str, Any]
    closure: Dict[str, Any]
    flap: Dict[str, Any]


class TriageFollowUp(BaseModel):
    pathology_review: str
    wound_care: str
    surveillance: str


class ProtocolInfo(BaseModel):
    """Information about guidelines/protocols applied in planning."""
    name: str = Field(..., description="Protocol bundle name")
    version: str = Field("0.1", description="Protocol/knowledge version")
    sources: List[str] = Field(default_factory=list, description="Primary sources/ids used")
    decisions: Dict[str, Any] = Field(default_factory=dict, description="Key decisions and gates applied")


class TriageResponse(BaseModel):
    diagnostics: TriageDiagnostics
    oncology: TriageOncology
    reconstruction: TriageReconstruction
    safety: TriageSafety
    overlay: TriageOverlay
    follow_up: TriageFollowUp
    citations: List[str] = Field(default_factory=list)
    artifacts: Optional[Artifacts] = None
    request_id: Optional[str] = None
    protocol: Optional[ProtocolInfo] = None
    site: Optional[str] = None
    rstl_angle_deg: Optional[float] = None
