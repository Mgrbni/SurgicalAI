# surgicalai/cli.py
import argparse, json, sys, time, os, glob, platform, webbrowser
from pathlib import Path
import numpy as np
import cv2

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

from .report import build_pdf


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _find_default_image(repo_root: Path) -> Path:
    candidates = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        candidates += glob.glob(str(repo_root / "data" / "lesions_sample" / ext))
    if not candidates:
        raise FileNotFoundError("No sample image found in data/lesions_sample/")
    return Path(candidates[0])


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def _face_center_mediapipe(img: np.ndarray):
    """Return approximate midline x using MediaPipe FaceMesh if available."""
    if not MP_AVAILABLE:
        return None

    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, refine_landmarks=False, max_num_faces=1) as fm:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        h, w = img.shape[:2]
        # Use nose bridge area (landmark 1 or 168 often near midline); if missing, average across several.
        idxs = [1, 4, 6, 168, 197]
        xs = []
        for i in idxs:
            if i < len(lm.landmark):
                xs.append(lm.landmark[i].x * w)
        if not xs:
            return None
        return float(np.mean(xs))


def _compute_asymmetry(img: np.ndarray, mid_x: float | None = None, thresh: float = 0.30):
    """Compute simple left-right asymmetry:
    1) Gray normalize
    2) Horizontal flip around estimated midline
    3) Per-pixel absolute difference -> heatmap + overlay
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray_f = gray.astype(np.float32) / 255.0

    if mid_x is None:
        mid_x = w / 2.0

    # Build a mirrored image around mid_x by shifting-flipping
    # Create coords for each pixel's mirror column
    xs = np.arange(w, dtype=np.float32)
    mirror_xs = (2 * mid_x - xs)
    mirror_xs = np.clip(mirror_xs, 0, w - 1)
    # Remap columns
    map_x = np.tile(mirror_xs, (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
    mirrored = cv2.remap(gray_f, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    diff = np.abs(gray_f - mirrored)
    diff = cv2.GaussianBlur(diff, (0, 0), 1.0)

    # Normalize to 0..1
    if diff.max() > 0:
        norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    else:
        norm = diff

    mask = (norm >= thresh).astype(np.uint8)
    area_pct = float(100.0 * mask.mean())
    asymmetry_index = float(norm.mean())

    # Heatmap (jet)
    heat_uint8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(img, 0.65, heat_color, 0.35, 0)

    # Draw midline
    x0 = int(round(mid_x))
    cv2.line(overlay, (x0, 0), (x0, h - 1), (0, 255, 255), 1)

    # Tiny metrics text in corner
    txt = f"asymmetry_index={asymmetry_index:.3f}  area={area_pct:.1f}%  thresh={thresh:.2f}"
    cv2.rectangle(overlay, (8, 8), (8 + 420, 36), (0, 0, 0), -1)
    cv2.putText(overlay, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return norm, heat_color, overlay, {"asymmetry_index": asymmetry_index, "area_pct": area_pct, "threshold": thresh}


def _open_path(p: Path):
    try:
        if platform.system() == "Darwin":
            os.system(f'open "{p}"')
        elif platform.system() == "Windows":
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            os.system(f'xdg-open "{p}" >/dev/null 2>&1 || true')
    except Exception:
        try:
            webbrowser.open_new_tab(str(p))
        except Exception:
            pass


def run_demo(args):
    t0 = time.time()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out).resolve()
    _ensure_dir(out_dir)

    # Input
    img_path = Path(args.input) if args.input else _find_default_image(repo_root)
    img = _read_image(img_path)

    # Midline estimate
    mid_x = _face_center_mediapipe(img) if MP_AVAILABLE else None

    # Compute asymmetry
    norm, heat_color, overlay, metrics = _compute_asymmetry(img, mid_x=mid_x, thresh=args.threshold)

    # Save artifacts
    input_copy = out_dir / "input.png"
    heat_path = out_dir / "heatmap.png"
    overlay_path = out_dir / "overlay.png"
    metrics_path = out_dir / "metrics.json"
    cv2.imwrite(str(input_copy), img)
    cv2.imwrite(str(heat_path), heat_color)
    cv2.imwrite(str(overlay_path), overlay)

    metrics_out = {
        "source_image": str(img_path),
        "asymmetry_index": metrics["asymmetry_index"],
        "area_pct": metrics["area_pct"],
        "threshold": metrics["threshold"],
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "timestamps": {"started": t0, "completed": time.time()},
        "runtime_sec": round(time.time() - t0, 3),
        "engine": {
            "mediapipe_face_mesh": bool(MP_AVAILABLE),
            "backend": "opencv",
            "mode": "cpu",
            "offline_llm": bool(args.offline_llm),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    # PDF
    pdf_path = out_dir / "report.pdf"
    build_pdf(
        pdf_path=pdf_path,
        title="SurgicalAI Asymmetry Report",
        subtitle="Photo → Landmarks (midline) → Asymmetry Heatmap",
        input_image=str(input_copy),
        overlay_image=str(overlay_path),
        metrics=metrics_out,
        footer="RESEARCH PROTOTYPE — NOT FOR CLINICAL USE",
    )

    # Optional: open outputs
    if not args.no_open:
        _open_path(out_dir)

    print(f"[OK] Wrote:\n  {overlay_path}\n  {heat_path}\n  {metrics_path}\n  {pdf_path}")
    print(f"[timing] total={metrics_out['runtime_sec']}s")


def main(argv=None):
    p = argparse.ArgumentParser(prog="surgicalai", description="SurgicalAI CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Run demo: image → heatmap → overlay → PDF")
    d.add_argument("--input", type=str, default=None, help="Path to input image (defaults to data/lesions_sample/*)")
    d.add_argument("--out", type=str, default="runs/demo", help="Output directory")
    d.add_argument("--cpu", action="store_true", help="Force CPU (no-op placeholder)")
    d.add_argument("--offline-llm", action="store_true", help="Run without network (flag only)")
    d.add_argument("--threshold", type=float, default=0.30, help="Asymmetry threshold for area%%")
    d.add_argument("--no-open", action="store_true", help="Do not auto-open output folder")
    d.set_defaults(func=run_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
