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
    lesion = lesion_model.classify(None)
    flap = flap_engine.suggest_flap({})
    visualizer.render(scan, lesion.heatmap, flap)
    logger.export_report({}, Path("report.pdf"))
    logger.save_metadata({}, Path("analysis.json"))
    return {
        "scan": scan,
        "landmarks": landmarks,
        "lesion": lesion,
        "flap": flap,
    }
