"""ROI-centered Grad-CAM and overlay generation with high-contrast visualization.

This module implements:
- Grad-CAM activation maps with ROI detection and centering
- High-contrast heatmap overlays with perceptual colormaps
- Dual artifact generation (full + zoomed overlays)
- Professional surgical annotation and contour visualization

NOT FOR CLINICAL USE - Demo purposes only.
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from skimage import measure, morphology, filters
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass

try:
    from .config import SETTINGS
except ImportError:
    # Fallback settings for when imported from server
    class _DummySettings:
        heatmap = {
            'roi_threshold': 0.35,
            'min_roi_area_px': 600,
            'roi_margin_pct': 0.18,
            'roi_out_size': 512,
            'heatmap_alpha': 0.55,
            'contour_levels': [0.3, 0.5, 0.7],
            'colormap': 'plasma',
            'gamma_correction': 1.2
        }
    SETTINGS = _DummySettings()


@dataclass
class ROIInfo:
    """Region of Interest information."""
    x: int
    y: int 
    w: int
    h: int
    center_x: int
    center_y: int
    area: int
    max_activation: float


def gradcam_activation(image: np.ndarray, model=None, layer: str = None) -> np.ndarray:
    """
    Generate Grad-CAM activation map for the given image.
    
    Args:
        image: RGB image as numpy array [H, W, 3]
        model: Model instance (placeholder for actual model)
        layer: Target layer name for Grad-CAM
        
    Returns:
        Activation map normalized to [0, 1] as float32 array [H, W]
    """
    # For demo purposes, create a realistic-looking activation map
    # In production, this would use actual Grad-CAM from the model
    
    h, w = image.shape[:2]
    
    # Create synthetic activation based on image content
    # Convert to grayscale and find dark regions (potential lesions)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Gaussian blur to create smooth activation regions
    blurred = cv2.GaussianBlur(gray, (51, 51), 20)
    
    # Invert so darker regions have higher activation
    activation = 255 - blurred
    
    # Add some noise and local maxima
    noise = np.random.RandomState(42).normal(0, 10, activation.shape)
    activation = activation.astype(np.float32) + noise
    
    # Create multiple local peaks
    for _ in range(3):
        center_x = np.random.RandomState(42).randint(w//4, 3*w//4)
        center_y = np.random.RandomState(42).randint(h//4, 3*h//4)
        
        # Create Gaussian peak
        y, x = np.ogrid[:h, :w]
        gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(w, h)//8)**2))
        activation += gaussian * 100
    
    # Normalize to [0, 1]
    activation = activation.astype(np.float32)
    if activation.max() > activation.min():
        activation = (activation - activation.min()) / (activation.max() - activation.min())
    else:
        activation = np.zeros_like(activation, dtype=np.float32)
    
    return activation


def find_lesion_roi(
    activation: np.ndarray, 
    threshold: float = 0.35,
    min_area: int = 600,
    margin_pct: float = 0.18
) -> ROIInfo:
    """
    Find the most significant region of interest from activation map.
    
    Args:
        activation: Activation map [0, 1] as float32 array [H, W]
        threshold: Binary threshold for ROI detection
        min_area: Minimum area in pixels for valid ROI
        margin_pct: Margin to add around ROI as percentage of ROI size
        
    Returns:
        ROIInfo with bounding box and center coordinates
    """
    h, w = activation.shape
    
    # Apply threshold with Otsu if no fixed threshold works
    try:
        if threshold <= 0:
            # Use Otsu threshold on the activation map
            activation_uint8 = (activation * 255).astype(np.uint8)
            threshold_val = filters.threshold_otsu(activation_uint8) / 255.0
        else:
            threshold_val = threshold
        
        # Create binary mask
        binary_mask = (activation > threshold_val).astype(np.uint8)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small objects
        binary_mask = morphology.remove_small_objects(
            binary_mask.astype(bool), min_size=min_area
        ).astype(np.uint8)
        
        # Fill holes
        binary_mask = morphology.remove_small_holes(
            binary_mask.astype(bool), area_threshold=min_area//2
        ).astype(np.uint8)
        
        # Find connected components
        labels = measure.label(binary_mask)
        regions = measure.regionprops(labels, intensity_image=activation)
        
        if not regions:
            # Fallback: use highest activation point
            max_y, max_x = np.unravel_index(np.argmax(activation), activation.shape)
            fallback_size = min(w, h) // 4
            x = max(0, max_x - fallback_size // 2)
            y = max(0, max_y - fallback_size // 2)
            w_roi = min(fallback_size, w - x)
            h_roi = min(fallback_size, h - y)
            
            return ROIInfo(
                x=x, y=y, w=w_roi, h=h_roi,
                center_x=x + w_roi//2, center_y=y + h_roi//2,
                area=w_roi * h_roi,
                max_activation=float(activation.max())
            )
        
        # Select best region (highest mean activation × area)
        best_region = max(regions, key=lambda r: r.mean_intensity * r.area)
        
        # Get bounding box
        min_row, min_col, max_row, max_col = best_region.bbox
        
        # Add margin
        roi_w = max_col - min_col
        roi_h = max_row - min_row
        margin_w = int(roi_w * margin_pct)
        margin_h = int(roi_h * margin_pct)
        
        # Expand with margin and clamp to image bounds
        x = max(0, min_col - margin_w)
        y = max(0, min_row - margin_h)
        x2 = min(w, max_col + margin_w)
        y2 = min(h, max_row + margin_h)
        
        roi_w_final = x2 - x
        roi_h_final = y2 - y
        
        return ROIInfo(
            x=x, y=y, w=roi_w_final, h=roi_h_final,
            center_x=x + roi_w_final//2, center_y=y + roi_h_final//2,
            area=roi_w_final * roi_h_final,
            max_activation=float(best_region.max_intensity)
        )
        
    except Exception as e:
        # Ultimate fallback: center of image
        fallback_size = min(w, h) // 3
        x = (w - fallback_size) // 2
        y = (h - fallback_size) // 2
        
        return ROIInfo(
            x=x, y=y, w=fallback_size, h=fallback_size,
            center_x=w//2, center_y=h//2,
            area=fallback_size * fallback_size,
            max_activation=float(activation.max()) if activation.size > 0 else 0.0
        )


def render_overlay(
    image: np.ndarray,
    activation: np.ndarray,
    alpha: float = 0.55,
    contour_levels: List[float] = [0.3, 0.5, 0.7],
    colormap: str = "plasma",
    center: Optional[Tuple[int, int]] = None,
    gamma: float = 1.2
) -> np.ndarray:
    """
    Render high-quality overlay with heatmap, contours, and annotations.
    
    Args:
        image: Base RGB image [H, W, 3]
        activation: Activation map [H, W] normalized to [0, 1]
        alpha: Heatmap opacity
        contour_levels: Activation levels for contour lines
        colormap: Matplotlib colormap name
        center: Lesion center coordinates for crosshair
        gamma: Gamma correction for base image
        
    Returns:
        Overlay image as RGB array [H, W, 3]
    """
    h, w = image.shape[:2]
    
    # Gamma correct the base image for better skin texture preservation
    base_image = image.copy().astype(np.float32) / 255.0
    base_image = np.power(base_image, 1.0/gamma)
    base_image = (base_image * 255).astype(np.uint8)
    
    # Create heatmap using perceptual colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(activation)  # Returns RGBA
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Create activation mask for selective blending
    activation_mask = activation > 0.1  # Only blend where there's significant activation
    
    # Blend heatmap with base image
    overlay = base_image.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            activation_mask,
            (1 - alpha) * base_image[:, :, c] + alpha * heatmap_rgb[:, :, c],
            base_image[:, :, c]
        )
    
    # Add contour lines
    overlay = _add_contour_lines(overlay, activation, contour_levels)
    
    # Add lesion center marker if provided
    if center:
        overlay = _add_center_marker(overlay, center)
    
    return overlay.astype(np.uint8)


def _add_contour_lines(
    image: np.ndarray, 
    activation: np.ndarray, 
    levels: List[float]
) -> np.ndarray:
    """Add anti-aliased contour lines at specified activation levels."""
    overlay = image.copy()
    
    for level in levels:
        # Find contours at this level
        binary = (activation > level).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours with anti-aliasing
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Skip tiny contours
                cv2.drawContours(overlay, [contour], -1, (255, 255, 255), 1, cv2.LINE_AA)
    
    return overlay


def _add_center_marker(image: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
    """Add crosshair and circle marker at lesion center."""
    overlay = image.copy()
    cx, cy = center
    
    # Crosshair
    cv2.line(overlay, (cx-15, cy), (cx+15, cy), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(overlay, (cx, cy-15), (cx, cy+15), (255, 255, 0), 2, cv2.LINE_AA)
    
    # Dotted circle
    radius = 12
    for angle in range(0, 360, 20):
        x1 = cx + int(radius * np.cos(np.radians(angle)))
        y1 = cy + int(radius * np.sin(np.radians(angle)))
        x2 = cx + int(radius * np.cos(np.radians(angle + 10)))
        y2 = cy + int(radius * np.sin(np.radians(angle + 10)))
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
    
    return overlay


def annotate_footer(
    image: np.ndarray,
    top_class: str,
    probability: float,
    roi_size: Tuple[int, int],
    timestamp: Optional[str] = None
) -> np.ndarray:
    """
    Add informative footer text to the overlay image.
    
    Args:
        image: Image to annotate
        top_class: Top predicted class
        probability: Probability of top class
        roi_size: (width, height) of the ROI
        timestamp: ISO timestamp string
        
    Returns:
        Annotated image
    """
    h, w = image.shape[:2]
    overlay = image.copy()
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()[:19]
    
    # Format footer text
    footer_text = f"Top: {top_class} • P={probability:.1%} • Created: {timestamp} • ROI: {roi_size[0]}×{roi_size[1]}px"
    
    # Text positioning and styling
    font_scale = 0.4
    thickness = 1
    padding = 8
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        footer_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Position at bottom-left with padding
    text_x = padding
    text_y = h - padding - baseline
    
    # Add semi-transparent background for better readability
    bg_x1 = text_x - 4
    bg_y1 = text_y - text_height - 4
    bg_x2 = text_x + text_width + 4
    bg_y2 = text_y + baseline + 4
    
    # Create semi-transparent overlay for text background
    text_bg = overlay.copy()
    cv2.rectangle(text_bg, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    overlay = cv2.addWeighted(overlay, 0.7, text_bg, 0.3, 0)
    
    # Add text with white color and slight stroke for contrast
    cv2.putText(overlay, footer_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Add subtle stroke for better contrast
    cv2.putText(overlay, footer_text, (text_x-1, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
    cv2.putText(overlay, footer_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return overlay


def save_overlays_full_and_zoom(
    image: np.ndarray,
    activation: np.ndarray,
    run_dir: Path,
    top_class: str,
    probability: float,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Path]:
    """
    Generate and save both full and zoomed overlay artifacts.
    
    Args:
        image: Original RGB image [H, W, 3]
        activation: Full-image activation map [H, W]
        run_dir: Directory to save artifacts
        top_class: Top predicted class name
        probability: Probability of top class
        config: Heatmap configuration dict
        
    Returns:
        Dictionary mapping artifact names to file paths
    """
    if config is None:
        config = getattr(SETTINGS, 'heatmap', {})
    
    # Get configuration with defaults
    roi_threshold = config.get('roi_threshold', 0.35)
    min_roi_area = config.get('min_roi_area_px', 600)
    roi_margin_pct = config.get('roi_margin_pct', 0.18)
    roi_out_size = config.get('roi_out_size', 512)
    # Support both 'alpha' (spec in settings.yaml) and legacy 'heatmap_alpha'
    heatmap_alpha = config.get('alpha', config.get('heatmap_alpha', 0.55))
    contour_levels = config.get('contour_levels', [0.3, 0.5, 0.7])
    colormap = config.get('colormap', 'plasma')
    gamma = config.get('gamma_correction', 1.2)
    
    # Find ROI
    roi_info = find_lesion_roi(activation, roi_threshold, min_roi_area, roi_margin_pct)
    
    # Generate full overlay
    full_overlay = render_overlay(
        image, activation, heatmap_alpha, contour_levels, colormap, 
        (roi_info.center_x, roi_info.center_y), gamma
    )
    
    # Add footer to full overlay
    full_overlay = annotate_footer(
        full_overlay, top_class, probability, (image.shape[1], image.shape[0])
    )
    
    # Crop to ROI and recompute activation for zoom
    roi_image = image[roi_info.y:roi_info.y+roi_info.h, 
                     roi_info.x:roi_info.x+roi_info.w]
    
    # Recompute activation on ROI crop for better alignment
    roi_activation = gradcam_activation(roi_image)
    
    # Resize ROI to target size
    roi_image_resized = cv2.resize(roi_image, (roi_out_size, roi_out_size), 
                                  interpolation=cv2.INTER_CUBIC)
    roi_activation_resized = cv2.resize(roi_activation, (roi_out_size, roi_out_size), 
                                       interpolation=cv2.INTER_CUBIC)
    
    # Generate zoom overlay
    zoom_center = (roi_out_size//2, roi_out_size//2)
    zoom_overlay = render_overlay(
        roi_image_resized, roi_activation_resized, heatmap_alpha, 
        contour_levels, colormap, zoom_center, gamma
    )
    
    # Add footer to zoom overlay
    zoom_overlay = annotate_footer(
        zoom_overlay, top_class, probability, (roi_info.w, roi_info.h)
    )
    
    # Save artifacts
    full_path = run_dir / "overlay_full.png"
    zoom_path = run_dir / "overlay_zoom.png"
    
    Image.fromarray(full_overlay).save(full_path)
    Image.fromarray(zoom_overlay).save(zoom_path)
    
    # Also save legacy overlay.png as the zoom version for backwards compatibility
    legacy_path = run_dir / "overlay.png"
    Image.fromarray(zoom_overlay).save(legacy_path)
    
    return {
        "overlay_full": full_path,
        "overlay_zoom": zoom_path,
        "overlay": legacy_path  # Legacy name
    }


def create_unified_heatmap(
    image: np.ndarray,
    activation: np.ndarray,
    run_dir: Path,
    colormap: str = "plasma"
) -> Path:
    """
    Create a standalone heatmap visualization without overlay.
    
    Args:
        image: Original RGB image
        activation: Activation map
        run_dir: Output directory
        colormap: Matplotlib colormap name
        
    Returns:
        Path to saved heatmap
    """
    # Create pure heatmap visualization
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(activation)
    heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    
    heatmap_path = run_dir / "heatmap.png"
    Image.fromarray(heatmap_rgb).save(heatmap_path)
    
    return heatmap_path
