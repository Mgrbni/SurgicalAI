from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    feats_dir: str = "feats"
    db_path: str = "lesions.duckdb"
    neighbors_k: int = 6
    demo_face_path: str = "data/samples/face.jpg"
    demo_lesion_path: str = "data/samples/lesion.jpg"
    report_title: str = "SurgicalAI Tier‑0 Triage Report"
    version: str = "tier0-0.1"

SETTINGS = Settings()
