"""High-level pipeline tying together facial analysis modules."""
from pathlib import Path
from typing import Any, Dict

from .scan_ingestion import ScanIngestor
from .landmark_detection import LandmarkDetector
from .lesion_detection import LesionDetector
from .flap_design_engine import FlapDesignEngine
from .visualization import SurgicalVisualizer
from .output_logging import OutputLogger


def run_pipeline(scan_path: Path) -> Dict[str, Any]:
    """Execute the facial reconstruction analysis pipeline."""
    ingestor = ScanIngestor()
    detector = LandmarkDetector()
    lesion_model = LesionDetector()
    flap_engine = FlapDesignEngine()
    visualizer = SurgicalVisualizer()
    logger = OutputLogger()

    scan = ingestor.load_scan(scan_path)
    landmarks = detector.detect_landmarks(scan)
    tension = detector.map_tension_lines(scan)
    anatomy = detector.overlay_anatomy(scan)

    lesion = lesion_model.classify(None)
    heatmap_3d = lesion_model.map_heatmap_to_3d(lesion.heatmap, scan)

    flap = flap_engine.suggest_flap(
        {"landmarks": landmarks, "tension": tension, "lesion": lesion}
    )

    visualizer.render(scan, heatmap_3d, flap, anatomy)

    logger.export_report({}, Path("report.pdf"))
    logger.save_metadata({}, Path("analysis.json"))
    logger.export_dicom({}, Path("analysis.dcm"))

    return {
        "scan": scan,
        "landmarks": landmarks,
        "tension": tension,
        "lesion": lesion,
        "flap": flap,
    }
