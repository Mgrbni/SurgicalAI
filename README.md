# SurgicalAI (minimal demo)

Quickstart

1) Install

```bash
python -m pip install -e ".[dev]"
```

2) One-line Demo

```bash
surgicalai demo assets/demo_face.jpg --out out
```

3) What it outputs

- Saves out/heatmap.png and out/overlay.png
- Prints a final JSON line with keys: ok, diagnosis_top3, probs, bbox, rstl_angle_deg, flap_candidates, artifacts

4) Determinism

boot_seed(1337) seeds Python/NumPy/Torch and disables cuDNN benchmark for repeatable results.

5) Tests

```bash
pytest -q
```

6) Limitations

Research demo only; not a medical device.
