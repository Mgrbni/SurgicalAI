"""UI pipeline runner."""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

from surgicalai.io.dermatology_loader import load_derm_image
from surgicalai.io.polycam_loader import load_mesh
from surgicalai.vision.models import load_model, predict
from surgicalai.vision.gradcam import compute_gradcam
from surgicalai.cognition.guidelines import load_rules, query_rules
from surgicalai.cognition.contraindications import GuidelineGate
from surgicalai.viz.overlays import save_overlay
from surgicalai.viz.guideline_card import GuidelineCard
from surgicalai.report.compose import make_pdf

def run_demo_ui(job_dir: Path, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run the demo pipeline for web UI.
    
    Args:
        job_dir: Directory to save artifacts
        args: Dict containing:
            image_path: Path to lesion image
            mesh_path: Optional path to Polycam mesh
            age: Optional patient age
            sex: Patient sex
            site: Facial subunit
            prior_histology: yes/no/unknown
            ill_defined_borders: bool
            recurrent: bool
            
    Returns:
        Dict containing analysis results and artifact paths
    """
    # 1. Load inputs
    img_rgb = load_derm_image(args["image_path"])
    
    mesh_data = None  # 3D disabled
    
    # 2. Run vision model
    model = load_model()
    probs = predict(model, img_rgb)
    
    # 3. Generate Grad-CAM
    gradcam = compute_gradcam(model, img_rgb)
    
    # 4. Check guidelines
    gate = GuidelineGate()
    high_risk = []
    if args["ill_defined_borders"]:
        high_risk.append("ill_defined_borders")
    if args["recurrent"]:
        high_risk.append("recurrent")
        
    gate_decision = gate.check(
        diagnosis="melanoma",  # Start conservative
        subunit=args["site"],
        melanoma_prob=probs["melanoma"],
        high_risk_features=high_risk
    )
    
    # 5. Get guidelines
    rules = load_rules()
    guidelines = query_rules("melanoma", args["site"], rules)
    
    # 6. Generate visualizations
    save_overlay(img_rgb, gradcam, job_dir / "overlay.png")
    
    # Save raw heatmap
    heatmap_img = Image.fromarray((gradcam * 255).astype(np.uint8))
    heatmap_img.save(job_dir / "heatmap.png")
    
    # Generate guideline card
    card = GuidelineCard()
    card.render(
        diagnosis="melanoma",
        subunit=args["site"],
        margins=guidelines.margins.__dict__,
        gate_decision=gate_decision.__dict__,
        flaps=guidelines.flap_options,
        danger_notes=guidelines.danger_structures,
        citations=guidelines.citations,
        output_path=job_dir / "guideline_card.png"
    )
    
    # 7. Generate report
    make_pdf(
        pdf_path=job_dir / "report.pdf",
        meta={
            "age": args["age"],
            "sex": args["sex"],
            "site": args["site"],
            "prior_histology": args["prior_histology"]
        },
        model_output={
            "probs": probs,
            "gate": gate_decision.__dict__
        },
        guidelines=guidelines.__dict__,
        artifacts={
            "overlay": str(job_dir / "overlay.png"),
            "heatmap": str(job_dir / "heatmap.png"),
            "guideline_card": str(job_dir / "guideline_card.png")
        }
    )
    
    # 8. Build response
    return {
        "class_probs": probs,
        "gate": gate_decision.__dict__,
        "flap_suggestions": guidelines.flap_options,
        "risk_notes": [note["notes"] for note in guidelines.danger_structures],
        "artifacts": {
            "overlay_png": str(job_dir / "overlay.png"),
            "heatmap_png": str(job_dir / "heatmap.png"),
            "guideline_card_png": str(job_dir / "guideline_card.png"),
            "report_pdf": str(job_dir / "report.pdf"),
            "report_json": str(job_dir / "report.json")
        },
        "citations_expanded": guidelines.citations
    }
