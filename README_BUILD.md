# SurgicalAI Build Instructions

## Build
```
scripts/build_win.bat
```
This script installs dependencies, runs linting and tests, and creates a
PyInstaller one-file executable under `dist/`.

## Smoke Test
```
scripts/smoke_test_win.bat
```
Runs the compiled executable on a synthetic mesh and validates the expected
artifacts.

Troubleshooting: ensure that `torch` and `torchvision` wheels are available for
the target platform. If PyInstaller reports missing DLLs, add `--collect-all`
for the offending package in `surgicalai.spec`.
