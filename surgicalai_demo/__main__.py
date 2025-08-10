import argparse
from pathlib import Path
from surgicalai_demo.pipeline import run_demo

def main():
    ap = argparse.ArgumentParser(description="SurgicalAI Tier‑0 demo (offline).")
    ap.add_argument("--image", type=str, default="", help="Path to lesion image (optional; if omitted, a picker opens)")
    ap.add_argument("--out", default="runs/demo", help="Output folder")
    ap.add_argument("--ask", action="store_true", help="Prompt for patient info (age/sex/site/photo type)")
    args = ap.parse_args()

    img = Path(args.image) if args.image else None
    run_demo(face_img_path=None, lesion_img_path=img, out_dir=Path(args.out), ask=args.ask)

if __name__ == "__main__":
    main()
