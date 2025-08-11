"""Langer's lines / RSTL overlay utilities.

Approach:
1. Estimate face orientation using simple heuristic (image aspect + vertical symmetry) to decide rotation.
2. Generate parametric bezier/curve approximations for key subregions (forehead, cheek, nose, chin) relative to image size.
3. Render semi-transparent poly-lines with optional suggested incision orientation given a defect center.

This is a lightweight placeholder — in production you would use a landmark detector (e.g., dlib/mediapipe) and map anatomical atlases.
"""
from __future__ import annotations
from typing import Tuple, Optional
from pathlib import Path
from math import sin, cos, pi
from PIL import Image, ImageDraw

LINE_COLOR = (20,120,200,140)
SUGGEST_COLOR = (220,40,40,200)


def _estimate_orientation(im: Image.Image) -> float:
    """Return rotation angle (deg) to make face upright (placeholder heuristic)."""
    w,h = im.size
    if h >= w:
        return 0.0
    # Landscape often indicates rotated selfie; rotate -90
    return -90.0


def _curve_points(w: int, h: int, region: str) -> list[tuple[int,int]]:
    pts: list[tuple[int,int]] = []
    if region == 'forehead':
        for t in range(0,101,5):
            x = int(w*0.15 + (w*0.7)*(t/100))
            y = int(h*0.18 + 10*sin(pi*t/100))
            pts.append((x,y))
    elif region == 'cheek_left':
        for t in range(0,101,5):
            x = int(w*0.28 - (w*0.15)*(t/100))
            y = int(h*0.30 + (h*0.28)*(t/100)**0.9)
            pts.append((x,y))
    elif region == 'cheek_right':
        for t in range(0,101,5):
            x = int(w*0.72 + (w*0.15)*(t/100))
            y = int(h*0.30 + (h*0.28)*(t/100)**0.9)
            pts.append((x,y))
    elif region == 'chin':
        for t in range(0,101,5):
            x = int(w*0.35 + (w*0.3)*(t/100))
            y = int(h*0.78 + 6*sin(pi*t/100))
            pts.append((x,y))
    return pts


def draw_langer_lines(image_path: Path, output_path: Path, lesion_center: Optional[Tuple[int,int]] = None, incision_angle: Optional[float] = None) -> Path:
    im = Image.open(image_path).convert('RGBA')
    rot = _estimate_orientation(im)
    if abs(rot) > 1e-2:
        im = im.rotate(rot, expand=True)
    overlay = Image.new('RGBA', im.size, (0,0,0,0))
    dr = ImageDraw.Draw(overlay)
    w,h = im.size
    for region in ['forehead','cheek_left','cheek_right','chin']:
        pts = _curve_points(w,h,region)
        if len(pts) > 1:
            dr.line(pts, fill=LINE_COLOR, width=2)
    # Suggested incision orientation: draw a short line through lesion center
    if lesion_center:
        x,y = lesion_center
        if incision_angle is None:
            incision_angle = 0.0
        length = int(min(w,h)*0.15)
        dx = length * cos(incision_angle*pi/180)/2
        dy = length * sin(incision_angle*pi/180)/2
        dr.line([(x-dx,y-dy),(x+dx,y+dy)], fill=SUGGEST_COLOR, width=4)
        dr.ellipse([x-6,y-6,x+6,y+6], fill=SUGGEST_COLOR)
    composed = Image.alpha_composite(im, overlay)
    composed.convert('RGB').save(output_path)
    return output_path
