"""End-to-end pipeline orchestration."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from ..io.polycam_loader import load_scan
from ..utils.rendering import render_views
from ..vision.classifier import LesionClassifier, LABELS
from ..vision.gradcam import GradCAM
from ..pipeline.lesion_localization import backproject_heatmap
from ..utils.io import save_json
from ..config import CONFIG


class Pipeline:
    def __init__(self):
        weights = CONFIG.models / 'resnet50_melanoma.pt'
        self.classifier = LesionClassifier(weights)
        self.gradcam = GradCAM(self.classifier.model)

    def run(self, scan_path: Path, photo_path: Path | None = None, run_id: str = 'demo'):
        data = load_scan(scan_path)
        vertices, faces = data['vertices'], data['faces']
        renders, cams = render_views(vertices, faces, img_size=CONFIG.img_size, views=1)
        img = Image.open(photo_path) if photo_path and Path(photo_path).exists() else Image.new('RGB',(224,224),(128,128,128))
        probs = self.classifier.predict(img)
        pred_idx = int(np.argmax(list(probs.values())))
        heatmap = self.gradcam(img, pred_idx)
        heat_vals = backproject_heatmap(vertices, heatmap)
        out_dir = CONFIG.outputs / run_id
        save_json({'probs': probs}, out_dir / 'summary.json')
        return {'vertices': vertices, 'heat': heat_vals, 'heatmap': heatmap, 'probs': probs, 'run_id': run_id}
