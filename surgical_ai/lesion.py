"""Skin lesion classification and heatmap generation."""

from typing import Any


class LesionDetector:
    """Classifies skin lesions and produces Grad-CAM heatmaps."""

    def classify(self, image: Any) -> dict[str, float]:
        """Return probabilistic diagnosis for ``image``.

        Parameters
        ----------
        image: Any
            2D projection or UV map of the facial scan.

        Returns
        -------
        dict
            Example: ``{"melanoma": 0.51, "benign": 0.49}``
        """
        # TODO: load pretrained CNN and perform inference
        raise NotImplementedError

    def generate_heatmap(self, image: Any) -> Any:
        """Generate Grad-CAM style heatmap for ``image``."""
        # TODO: implement Grad-CAM using torchcam or captum
        raise NotImplementedError
