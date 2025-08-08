# SurgicalAI

Modular framework for cognitive and contraindication-aware surgical assistance.

## Modules

- **Cognitive Decision Engine**: evaluates flap feasibility.
- **Contraindication Detector**: flags anatomical or systemic risks.
- **Structural Integrity Validator**: checks biomechanical soundness.
- **Self-Audit Checker**: ensures pipeline consistency.
- **Smart Report Generator**: produces summaries of decisions and warnings.

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
