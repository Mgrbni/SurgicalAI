"""Skin lesion classification and heatmap generation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LesionResult:
    """Probabilistic diagnosis and accompanying heatmap."""

    probabilities: Dict[str, float]
    heatmap: Optional[Any] = None


class LesionDetector:
    """Classifies skin lesions and produces Grad-CAM heatmaps."""

    def classify(self, image: Any) -> LesionResult:
        """Return probabilistic diagnosis for ``image``.

        Parameters
        ----------
        image:
            2D projection or UV map of the facial scan.

        Returns
        -------
        LesionResult
            Probabilities for classes and placeholder heatmap.
        """
        # TODO: load pretrained CNN and perform inference
        raise NotImplementedError

    def generate_heatmap(self, image: Any) -> Any:
        """Generate Grad-CAM style heatmap for ``image``."""
        # TODO: implement Grad-CAM using torchcam or captum
        raise NotImplementedError
