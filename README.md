# SurgicalAI

**Research prototype. Not for clinical use.**

## Quickstart
```bash
pip install -r requirements.txt
python -m surgicalai.demo --all
uvicorn surgicalai.api.server:app --reload
```

## Module Map
```
api/       FastAPI server
ui/        Gradio dashboard
pipeline/  Core analysis orchestration
planning/  Flap planning
risk/      Contraindication and biomechanics
viz/       Visualization helpers
```

## Troubleshooting
All assets are synthetic and run on CPU. Sample images are generated on the fly so no binary data is stored in the repository.
