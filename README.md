# SurgicalAI

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Research Prototype – Not for Clinical Use**

Minimal pipeline for reconstructive surgery research.

## Quickstart

```bash
git clone <repo> && cd SurgicalAI
make demo
```

Outputs appear in `runs/demo/` and open in a browser. The demo uses only synthetic data and runs entirely on CPU.

## CLI

```
surgicalai demo --input data/lesions_sample --cpu --offline-llm --out runs/demo
surgicalai train --data data/lesions_sample
surgicalai evaluate --checkpoint toy.pt
surgicalai api --host 0.0.0.0 --port 8000
surgicalai ui
```

Flags:
- `--offline-llm` (default) prevents any network calls.
- `--cpu` forces CPU execution.

## Docker (optional)

```bash
docker build -t ghcr.io/<YOURORG>/surgicalai:dev .
```

GHCR TBD.

## Privacy

See [docs/PRIVACY.md](docs/PRIVACY.md) for data flow and guarantees.

## Development

```bash
ruff check .
black --check .
mypy surgicalai
pytest -q
make demo
```

No binaries are committed; CI builds artifacts.

## License

Apache-2.0
