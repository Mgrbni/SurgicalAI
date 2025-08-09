"""Simple image classifier with optional Torch backend.

The real project is expected to use a ResNet-50 model from
``torchvision``.  For the unit tests in this kata we want the module to
remain lightweight and runnable even when ``torch`` is not installed.
The implementation therefore tries to import the deep learning stack
and falls back to a tiny NumPy-based stub that returns uniform
probabilities.  This keeps the public API identical while avoiding a
heavy dependency during tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from PIL import Image
import numpy as np

try:  # pragma: no cover - exercised in environments with torch
    import torch
    from torchvision import models, transforms
except Exception:  # pragma: no cover - torch not available
    torch = None
    models = None
    transforms = None

from ..config import CONFIG

LABELS = ["benign", "melanoma", "nevus", "other"]


class LesionClassifier:
    """Wrapper around a classification model.

    When Torch is unavailable a minimal stub is used which mimics the
    interface of a trained network.  The stub simply outputs equal
    probabilities for all classes.
    """

    def __init__(self, weights_path: Path | None = None):
        if torch is not None:
            self.model = models.resnet50(weights=None)
            self.model.eval()
            if weights_path and weights_path.exists():
                self.model.load_state_dict(
                    torch.load(weights_path, map_location="cpu")
                )
            self.tf = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor()]
            )
        else:  # fallback stub
            self.model = None
            self.tf = lambda img: np.asarray(img.resize((224, 224))) / 255.0

    def predict(self, image: Image.Image) -> Dict[str, float]:
        """Return class probabilities for ``image``."""

        if torch is not None and self.model is not None:
            x = self.tf(image).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        else:  # simple uniform probabilities
            _ = self.tf(image)  # ensure resize for consistent behaviour
            probs = np.full(len(LABELS), 1.0 / len(LABELS))
        return {label: float(probs[i]) for i, label in enumerate(LABELS)}
