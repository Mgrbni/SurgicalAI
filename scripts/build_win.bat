#!/usr/bin/env bash
set -e
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
export PYTHONPATH=.
ruff check surgicalai_cli.py surgicalai tests
pytest -q
mkdir -p dist
cp surgicalai_cli.py dist/SurgicalAI.exe
sha256sum dist/SurgicalAI.exe
