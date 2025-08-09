# surgicalai

Research prototype for surgical planning and visualization.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .
```

Run demo:

```bash
surgicalai demo
```

Run API/UI:

```bash
surgicalai ingest
surgicalai analyze
surgicalai plan
surgicalai visualize
```

## Architecture

```
surgicalai
├── surgicalai/        # core package
│   ├── cli.py
│   ├── config.py
│   └── utils/
├── models/            # model weights
└── data/
    ├── samples/
    └── anatomy/
```

## Medical Disclaimer

Research prototype. Not for clinical use.

## Troubleshooting

- Ensure dependencies are installed with `pip install -r requirements.txt`.
- Use `python -m pip install --upgrade pip` if installations fail.
- For rendering issues, verify Open3D support on your platform.
