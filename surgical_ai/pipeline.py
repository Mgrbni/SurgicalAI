"""High level orchestration of the SurgicalAI modules.

The :class:`SurgicalPipeline` class demonstrates the expected flow of data
through the system. Each step delegates to the corresponding submodule and is
currently a placeholder awaiting real implementations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .flap import FlapDesigner, FlapPlan
from .landmarks import LandmarkDetector, LandmarkResult
from .lesion import LesionDetector, LesionResult
from .scan import ScanData, ScanProcessor


@dataclass
class PipelineResult:
    """Aggregate outputs from the full pipeline."""

    scan: ScanData
    landmarks: LandmarkResult
    lesion: LesionResult
    flap: FlapPlan


class SurgicalPipeline:
    """Orchestrates individual modules to produce a surgical plan."""

    def __init__(
        self,
        scan_processor: Optional[ScanProcessor] = None,
        landmark_detector: Optional[LandmarkDetector] = None,
        lesion_detector: Optional[LesionDetector] = None,
        flap_designer: Optional[FlapDesigner] = None,
    ) -> None:
        self.scan_processor = scan_processor or ScanProcessor()
        self.landmark_detector = landmark_detector or LandmarkDetector()
        self.lesion_detector = lesion_detector or LesionDetector()
        self.flap_designer = flap_designer or FlapDesigner()

    def run(self, path: str | Path) -> PipelineResult:
        """Execute the pipeline on a Polycam scan located at ``path``.

        This method outlines the interactions between modules but leaves actual
        implementations to future development.
        """

        scan = self.scan_processor.load_scan(path)
        scan = self.scan_processor.normalize(scan)

        landmarks = self.landmark_detector.detect(scan)
        tension_map = landmarks.tension_map or self.landmark_detector.map_tension_lines(scan)

        # In a full implementation, the scan would be projected to 2D before
        # lesion analysis. Here we simply pass ``None`` as a placeholder.
        lesion = self.lesion_detector.classify(None)
        lesion.heatmap = self.lesion_detector.generate_heatmap(None)

        flap = self.flap_designer.suggest((0.0, 0.0, 0.0), tension_map)

        return PipelineResult(scan=scan, landmarks=landmarks, lesion=lesion, flap=flap)
