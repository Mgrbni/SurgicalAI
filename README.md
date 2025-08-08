# SurgicalAI

Modular framework for cognitive and contraindication-aware surgical assistance.

## Modules

- **Cognitive Decision Engine**: evaluates flap feasibility.
- **Contraindication Detector**: flags anatomical or systemic risks.
- **Structural Integrity Validator**: checks biomechanical soundness.
- **Self-Audit Checker**: ensures pipeline consistency.
- **Smart Report Generator**: produces summaries of decisions and warnings.
- **Facial Analysis**: handles 3D scan ingestion, landmark detection, lesion analysis, flap design, visualization, and reporting.

## Usage

```python
from surgical_ai.pipeline import run_pipeline

context = {
    "lesion": {"size": 4},
    "rotation_angle": 42,
    "lesion_zone": "left_cheek",
    "heatmap_zone": "left_cheek",
}
results = run_pipeline(context)
print(results["report"].summary)
```

### Facial Analysis Pipeline

```python
from pathlib import Path
from surgical_ai.facial_analysis import run_pipeline as facial_run

results = facial_run(Path("face_scan.ply"))
print(results["flap"].flap_type)
```

### Sample Facial Analysis Output

```
Diagnosis: 51% probability of melanoma
Lesion Coordinates (X=23.1, Y=48.7, Z=3.8)
Suggested Flap: Rotation flap from lateral cheek
Tension Zone: Zygomatic to mandibular angle
Predicted Success Rate: 93%
```
