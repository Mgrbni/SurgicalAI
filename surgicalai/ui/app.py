"""Gradio dashboard."""
from __future__ import annotations
import gradio as gr
import numpy as np
from pathlib import Path
from ..pipeline.integration import Pipeline
from ..planning.flap_designer import design_flap
from ..risk.contraindications import assess_risk
from ..risk.biomech_validator import estimate_strain, validate_strain
from ..viz.overlays import overlay_heatmap
from PIL import Image
from ..config import CONFIG

pipe = Pipeline()

def run_pipeline(scan_path, photo_path):
    res = pipe.run(Path(scan_path.name), Path(photo_path.name) if photo_path else None, run_id='ui')
    flap = design_flap(res['vertices'], res['heat'])
    risk = assess_risk(res['vertices'][0])
    strain = validate_strain(estimate_strain(res['vertices'], res['vertices'], np.array(flap['vector'])))
    img = Image.open(photo_path.name) if photo_path else Image.new('RGB',(224,224),(128,128,128))
    heat_img = overlay_heatmap(img, res['heatmap'])
    return res['probs'], flap, risk, strain, heat_img

css = """#header{background-color:#222;color:white;text-align:center;padding:10px}"""
with gr.Blocks(css=css) as demo:
    gr.Markdown('Research prototype. Not for clinical use.')
    gr.Markdown("<div id='header'><h1>SurgicalAI Demo</h1></div>")
    with gr.Row():
        scan = gr.File(label='Scan OBJ')
        photo = gr.File(label='Photo')
    run_btn = gr.Button('Run')
    probs = gr.JSON(label='Probabilities')
    flap = gr.JSON(label='Flap')
    risk = gr.JSON(label='Risk')
    strain = gr.JSON(label='Strain')
    img_out = gr.Image(label='Overlay')
    run_btn.click(run_pipeline, inputs=[scan, photo], outputs=[probs, flap, risk, strain, img_out])

def launch():
    demo.launch()
