#!/usr/bin/env bash
set -e
python - <<'PY'
import pathlib, trimesh
p = pathlib.Path('samples/synthetic_face.obj')
p.parent.mkdir(exist_ok=True)
if not p.exists():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    mesh.export(p)
PY
PYTHONPATH=. python dist/SurgicalAI.exe run --input samples/synthetic_face.obj --output-dir build_artifacts/synthetic --device cpu --report html
art_dir=build_artifacts/synthetic
required=(features.npz landmarks.json lesion_probs.json heatmap_overlay.png flap_overlay.png report.html flap_plan.json validator.json)
missing=0
for f in "${required[@]}"; do
    [ -f "$art_dir/$f" ] || missing=1
done
hash=$(sha256sum dist/SurgicalAI.exe | awk '{print $1}')
if [ $missing -eq 0 ]; then
    status=passed
else
    status=failed
fi
python - <<PY
import json,glob
print(json.dumps({
    'smoke_test': '$status',
    'artifacts': glob.glob('$art_dir/*'),
    'exe_sha256': '$hash',
    'exit_code': 0 if '$status'=='passed' else 1
}))
PY
if [ "$status" = "passed" ]; then exit 0; else exit 1; fi
