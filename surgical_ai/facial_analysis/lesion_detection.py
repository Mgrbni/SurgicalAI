"""Skin lesion detection using CNN models."""
from dataclasses import dataclass
from typing import Any


@dataclass
class LesionAnalysis:
    """Result of lesion classification."""

    probability: float
    heatmap: Any


class LesionDetector:
    """Classifies lesions and produces heatmaps."""

    def classify(self, image: Any) -> LesionAnalysis:
        """Run lesion classification model.

        Parameters
        ----------
        image: Any
            2D projection or UV map derived from scan.

        Returns
        -------
        LesionAnalysis
            Probability and heatmap placeholder.
        """
        # Placeholder: run model and generate Grad-CAM.
        return LesionAnalysis(probability=0.0, heatmap=None)
