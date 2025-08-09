# SurgicalAI

**Research Prototype – Not for Clinical Use**

Minimal pipeline demonstrating analysis, planning and visualization for reconstructive surgery.

## One Command Demo

```bash
pip install -r requirements.txt
pip install -e .

surgicalai demo --out outputs/demo
```

To include an OpenAI generated narrative, set `OPENAI_API_KEY` and run:

```bash
surgicalai demo --out outputs/demo --with-llm --model gpt-4o
```

Outputs are written to the specified folder and include meshes, JSON files, images and a PDF report.

## Train / Evaluate / Retrain

```bash
surgicalai dataset sync --dsn <DSN> --root data/lesions
surgicalai train --data-source folder --root data/lesions --epochs 10
surgicalai evaluate --checkpoint models/resnet50_best.pt
surgicalai retrain --from-csv outputs/train/misclassified.csv --epochs 3
```

## Package EXE

```bash
surgicalai package
dist/SurgicalAI.exe --demo --out outputs/demo_exe
```

## DB (optional)

Database DSN template:

```
postgresql+psycopg2://user:pass@host/dbname
```

Tables: `images` (id, uri_or_path, label, split, verified, notes, created_at, updated_at) and `predictions` (id, image_id, label, prob, gradcam, created_at).

## API and UI

Start the API:

```bash
python -m surgicalai.api
```

Launch the Gradio UI:

```bash
python -m surgicalai.ui
```

## LLM Usage

Only de-identified numeric summaries are sent to the OpenAI API. Meshes and images never leave the machine.

## Development

Run tests with:

```bash
pytest -q
```

See [docs/IO_SCHEMA.md](docs/IO_SCHEMA.md) for data contracts.

## License

Apache-2.0
