from __future__ import annotations

from pathlib import Path
import tempfile
import gradio as gr

from surgicalai.demo import run as demo_run


def _run(mesh_file, use_llm):
    out = Path(tempfile.mkdtemp())
    if mesh_file is not None:
        # Not implemented: using synthetic mesh
        pass
    demo_run(out, with_llm=use_llm)
    overview = out / "overview.png"
    report = out / "report.pdf"
    narrative = (out / "narrative.txt").read_text() if (out / "narrative.txt").exists() else ""
    return overview, report, narrative


def launch() -> None:  # pragma: no cover
    iface = gr.Interface(
        fn=_run,
        inputs=[gr.File(label="Mesh"), gr.Checkbox(label="Use GPT summary")],
        outputs=[gr.Image(label="Overview"), gr.File(label="Report"), gr.Textbox(label="Narrative")],
        title="SurgicalAI",
    )
    iface.launch()


if __name__ == "__main__":  # pragma: no cover
    launch()
