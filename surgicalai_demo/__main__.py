import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from surgicalai_demo.pipeline import run_demo

def main():
    ap = argparse.ArgumentParser(description="SurgicalAI Tier‑0 demo (offline).")
    ap.add_argument("--samples_dir", default="data/samples", help="Folder with demo images")
    ap.add_argument("--out", default="runs/demo", help="Output folder")
    args = ap.parse_args()

    samples = Path(args.samples_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def ensure_samples(samples_dir: Path):
        samples_dir.mkdir(parents=True, exist_ok=True)
        face_img = samples_dir / "face.jpg"
        lesion_img = samples_dir / "lesion.jpg"

        if not face_img.exists():
            img = Image.new("RGB", (256, 256), (224, 172, 105))
            img.save(face_img)

        if not lesion_img.exists():
            img = Image.new("RGB", (256, 256), (255, 224, 189))
            draw = ImageDraw.Draw(img)
            draw.ellipse((80, 80, 176, 176), fill=(90, 50, 30))
            img.save(lesion_img)

        return face_img, lesion_img

    face_img, lesion_img = ensure_samples(samples)
    run_demo(face_img, lesion_img, out)

if __name__ == "__main__":
    main()
