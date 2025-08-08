# SurgicalAI

Created by Dr. Mehdi Ghorbanikarimabad

## Prototype Structure

This repository contains a modular Python package for exploring
reconstructive plastic surgery workflows.

### Modules

- `scan` – 3D facial scan ingestion and normalization
- `landmarks` – anatomical landmark detection and tension line mapping
- `lesion` – skin lesion classification with Grad-CAM heatmaps
- `flap` – local flap design suggestion engine
- `visualization` – 3D rendering of scans, lesions and flaps
- `output` – report generation and metadata export
- `pipeline` – orchestrates the above modules into a single workflow

Each module currently provides placeholders for future implementation.

```python
from surgical_ai.pipeline import SurgicalPipeline

pipeline = SurgicalPipeline()
# pipeline.run("/path/to/scan.obj")  # raises NotImplementedError until filled in
```
