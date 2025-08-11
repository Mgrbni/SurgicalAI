"""RSTL overlay visualization with SVG rendering."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

from ..geometry.rstl import RSTLLine

class RSTLOverlayRenderer:
    """Renders RSTL overlays on images using SVG and raster methods."""
    
    def __init__(self):
        self.default_colors = {
            'forehead': '#FF6B6B',
            'glabella': '#4ECDC4', 
            'left_periorbital': '#45B7D1',
            'right_periorbital': '#45B7D1',
            'nasal_dorsum': '#96CEB4',
            'nasal_tip': '#FFEAA7',
            'left_cheek': '#DDA0DD',
            'right_cheek': '#DDA0DD',
            'perioral': '#98D8C8',
            'chin': '#F7DC6F',
            'mandibular': '#BB8FCE',
            'default': '#74B9FF'
        }
    
    def render_rstl_overlay_raster(self, image: np.ndarray, 
                                  rstl_lines: List[RSTLLine],
                                  opacity: float = 0.6,
                                  line_thickness: int = 2) -> np.ndarray:
        """Render RSTL overlay directly on raster image."""
        
        overlay = image.copy()
        
        for line in rstl_lines:
            if line.confidence < 0.3:
                continue  # Skip low-confidence lines
            
            # Get color for region
            color_hex = self.default_colors.get(line.region, self.default_colors['default'])
            color_bgr = self._hex_to_bgr(color_hex)
            
            # Adjust line thickness based on confidence
            thickness = max(1, int(line_thickness * line.confidence))
            
            # Draw line
            start_pt = (int(line.start_point[0]), int(line.start_point[1]))
            end_pt = (int(line.end_point[0]), int(line.end_point[1]))
            
            cv2.line(overlay, start_pt, end_pt, color_bgr, thickness)
            
            # Draw direction arrow for longer lines
            line_length = np.sqrt((end_pt[0] - start_pt[0])**2 + (end_pt[1] - start_pt[1])**2)
            if line_length > 30:
                self._draw_direction_arrow(overlay, start_pt, end_pt, color_bgr, thickness)
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - opacity, overlay, opacity, 0)
        return result
    
    def render_rstl_overlay_svg(self, image_shape: Tuple[int, int], 
                               rstl_lines: List[RSTLLine],
                               output_path: Path,
                               opacity: float = 0.6) -> str:
        """Render RSTL overlay as SVG."""
        
        height, width = image_shape[:2]
        
        # Create SVG root element
        svg = ET.Element('svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f'0 0 {width} {height}')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # Create defs section for reusable elements
        defs = ET.SubElement(svg, 'defs')
        
        # Add arrowhead marker
        marker = ET.SubElement(defs, 'marker')
        marker.set('id', 'arrowhead')
        marker.set('markerWidth', '10')
        marker.set('markerHeight', '7')
        marker.set('refX', '9')
        marker.set('refY', '3.5')
        marker.set('orient', 'auto')
        
        polygon = ET.SubElement(marker, 'polygon')
        polygon.set('points', '0 0, 10 3.5, 0 7')
        polygon.set('fill', '#333')
        
        # Create group for RSTL lines
        rstl_group = ET.SubElement(svg, 'g')
        rstl_group.set('id', 'rstl-lines')
        rstl_group.set('opacity', str(opacity))
        
        # Group lines by region for better organization
        region_groups = {}
        for line in rstl_lines:
            if line.confidence < 0.3:
                continue
                
            if line.region not in region_groups:
                region_group = ET.SubElement(rstl_group, 'g')
                region_group.set('id', f'region-{line.region}')
                region_group.set('class', f'rstl-region rstl-{line.region}')
                region_groups[line.region] = region_group
        
        # Add lines to their respective region groups
        for line in rstl_lines:
            if line.confidence < 0.3:
                continue
            
            region_group = region_groups[line.region]
            
            # Get color for region
            color = self.default_colors.get(line.region, self.default_colors['default'])
            
            # Calculate line thickness based on confidence
            stroke_width = max(1, int(3 * line.confidence))
            
            # Create line element
            line_elem = ET.SubElement(region_group, 'line')
            line_elem.set('x1', f'{line.start_point[0]:.1f}')
            line_elem.set('y1', f'{line.start_point[1]:.1f}')
            line_elem.set('x2', f'{line.end_point[0]:.1f}')
            line_elem.set('y2', f'{line.end_point[1]:.1f}')
            line_elem.set('stroke', color)
            line_elem.set('stroke-width', str(stroke_width))
            line_elem.set('stroke-opacity', f'{line.confidence:.2f}')
            line_elem.set('stroke-linecap', 'round')
            
            # Add direction indicator for longer lines
            line_length = np.sqrt((line.end_point[0] - line.start_point[0])**2 + 
                                (line.end_point[1] - line.start_point[1])**2)
            if line_length > 40:
                line_elem.set('marker-end', 'url(#arrowhead)')
        
        # Add legend
        self._add_svg_legend(svg, width, height)
        
        # Save SVG
        svg_string = self._prettify_svg(svg)
        output_path.write_text(svg_string, encoding='utf-8')
        
        return svg_string
    
    def render_excision_alignment(self, image: np.ndarray,
                                 alignment_info: Dict[str, Any],
                                 color: Tuple[int, int, int] = (0, 255, 0),
                                 thickness: int = 3) -> np.ndarray:
        """Render optimal excision alignment on image."""
        
        overlay = image.copy()
        center = alignment_info['center']
        major_length = alignment_info['major_axis_length']
        minor_length = alignment_info['minor_axis_length']
        angle = alignment_info['rstl_alignment_angle']
        
        # Convert to OpenCV format
        center_cv = (int(center[0]), int(center[1]))
        axes = (int(major_length / 2), int(minor_length / 2))
        angle_cv = angle
        
        # Draw excision ellipse
        cv2.ellipse(overlay, center_cv, axes, angle_cv, 0, 360, color, thickness)
        
        # Draw center point
        cv2.circle(overlay, center_cv, 5, color, -1)
        
        # Draw RSTL alignment indicator
        rstl_direction = alignment_info['long_axis_direction']
        line_length = max(axes) + 20
        
        end_x = center[0] + rstl_direction[0] * line_length
        end_y = center[1] + rstl_direction[1] * line_length
        
        cv2.arrowedLine(overlay, center_cv, (int(end_x), int(end_y)), 
                       color, thickness, tipLength=0.3)
        
        # Add confidence indicator
        confidence = alignment_info.get('confidence', 0.5)
        confidence_color = self._confidence_to_color(confidence)
        cv2.putText(overlay, f'RSTL Confidence: {confidence:.2f}', 
                   (center_cv[0] + 20, center_cv[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, confidence_color, 2)
        
        return overlay
    
    def create_interactive_svg_overlay(self, image_shape: Tuple[int, int],
                                     rstl_lines: List[RSTLLine],
                                     alignment_info: Optional[Dict[str, Any]] = None,
                                     output_path: Path = None) -> str:
        """Create an interactive SVG overlay with hover effects and region highlighting."""
        
        height, width = image_shape[:2]
        
        # Create SVG with interactive features
        svg = ET.Element('svg')
        svg.set('width', str(width))
        svg.set('height', str(height))
        svg.set('viewBox', f'0 0 {width} {height}')
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        
        # Add CSS styles for interactivity
        style = ET.SubElement(svg, 'style')
        style.text = """
        .rstl-line {
            cursor: pointer;
            transition: stroke-width 0.2s ease;
        }
        .rstl-line:hover {
            stroke-width: 5px !important;
            stroke-opacity: 1.0 !important;
        }
        .region-label {
            font-family: Arial, sans-serif;
            font-size: 12px;
            fill: #333;
            text-anchor: middle;
            pointer-events: none;
        }
        .confidence-indicator {
            font-family: Arial, sans-serif;
            font-size: 10px;
            fill: #666;
        }
        .excision-guide {
            stroke-dasharray: 5,5;
            animation: dash 2s linear infinite;
        }
        @keyframes dash {
            to { stroke-dashoffset: -10; }
        }
        """
        
        # Create interactive RSTL lines
        for i, line in enumerate(rstl_lines):
            if line.confidence < 0.3:
                continue
            
            color = self.default_colors.get(line.region, self.default_colors['default'])
            stroke_width = max(2, int(4 * line.confidence))
            
            line_elem = ET.SubElement(svg, 'line')
            line_elem.set('class', f'rstl-line rstl-{line.region}')
            line_elem.set('x1', f'{line.start_point[0]:.1f}')
            line_elem.set('y1', f'{line.start_point[1]:.1f}')
            line_elem.set('x2', f'{line.end_point[0]:.1f}')
            line_elem.set('y2', f'{line.end_point[1]:.1f}')
            line_elem.set('stroke', color)
            line_elem.set('stroke-width', str(stroke_width))
            line_elem.set('stroke-opacity', f'{line.confidence:.2f}')
            line_elem.set('stroke-linecap', 'round')
            
            # Add tooltip
            title = ET.SubElement(line_elem, 'title')
            title.text = f'Region: {line.region.replace("_", " ").title()}\nConfidence: {line.confidence:.2f}'
        
        # Add excision alignment if provided
        if alignment_info:
            self._add_svg_excision_guide(svg, alignment_info)
        
        # Add region labels
        self._add_region_labels(svg, rstl_lines)
        
        if output_path:
            svg_string = self._prettify_svg(svg)
            output_path.write_text(svg_string, encoding='utf-8')
            return svg_string
        
        return self._prettify_svg(svg)
    
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to BGR tuple."""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
    
    def _draw_direction_arrow(self, image: np.ndarray, start: Tuple[int, int], 
                            end: Tuple[int, int], color: Tuple[int, int, int], 
                            thickness: int):
        """Draw direction arrow on line."""
        
        # Calculate arrow position (75% along the line)
        arrow_pos = (
            int(start[0] + 0.75 * (end[0] - start[0])),
            int(start[1] + 0.75 * (end[1] - start[1]))
        )
        
        # Calculate arrow direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
        
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Arrow size
        arrow_length = min(10, length * 0.2)
        
        # Arrow tip
        tip_x = int(arrow_pos[0] + dx_norm * arrow_length * 0.5)
        tip_y = int(arrow_pos[1] + dy_norm * arrow_length * 0.5)
        
        # Arrow wings
        wing_length = arrow_length * 0.4
        wing_angle = 2.5  # radians
        
        wing1_x = int(tip_x - dx_norm * wing_length * np.cos(wing_angle) + dy_norm * wing_length * np.sin(wing_angle))
        wing1_y = int(tip_y - dy_norm * wing_length * np.cos(wing_angle) - dx_norm * wing_length * np.sin(wing_angle))
        
        wing2_x = int(tip_x - dx_norm * wing_length * np.cos(wing_angle) - dy_norm * wing_length * np.sin(wing_angle))
        wing2_y = int(tip_y - dy_norm * wing_length * np.cos(wing_angle) + dx_norm * wing_length * np.sin(wing_angle))
        
        # Draw arrow
        arrow_points = np.array([(tip_x, tip_y), (wing1_x, wing1_y), (wing2_x, wing2_y)], np.int32)
        cv2.fillPoly(image, [arrow_points], color)
    
    def _confidence_to_color(self, confidence: float) -> Tuple[int, int, int]:
        """Convert confidence score to color (green=high, red=low)."""
        # Interpolate between red (low) and green (high)
        red = int(255 * (1 - confidence))
        green = int(255 * confidence)
        blue = 0
        return (blue, green, red)  # BGR format
    
    def _add_svg_legend(self, svg: ET.Element, width: int, height: int):
        """Add legend to SVG."""
        
        legend_group = ET.SubElement(svg, 'g')
        legend_group.set('id', 'legend')
        legend_group.set('transform', f'translate({width - 150}, 20)')
        
        # Legend background
        legend_bg = ET.SubElement(legend_group, 'rect')
        legend_bg.set('x', '0')
        legend_bg.set('y', '0') 
        legend_bg.set('width', '140')
        legend_bg.set('height', f'{len(self.default_colors) * 20 + 20}')
        legend_bg.set('fill', 'white')
        legend_bg.set('fill-opacity', '0.9')
        legend_bg.set('stroke', '#333')
        legend_bg.set('stroke-width', '1')
        
        # Legend title
        title = ET.SubElement(legend_group, 'text')
        title.set('x', '70')
        title.set('y', '15')
        title.set('text-anchor', 'middle')
        title.set('font-family', 'Arial, sans-serif')
        title.set('font-size', '12')
        title.set('font-weight', 'bold')
        title.text = 'RSTL Regions'
        
        # Legend items
        y_offset = 35
        for region, color in self.default_colors.items():
            if region == 'default':
                continue
                
            # Color square
            color_rect = ET.SubElement(legend_group, 'rect')
            color_rect.set('x', '10')
            color_rect.set('y', str(y_offset - 8))
            color_rect.set('width', '12')
            color_rect.set('height', '12')
            color_rect.set('fill', color)
            
            # Label
            label = ET.SubElement(legend_group, 'text')
            label.set('x', '28')
            label.set('y', str(y_offset + 2))
            label.set('font-family', 'Arial, sans-serif')
            label.set('font-size', '10')
            label.text = region.replace('_', ' ').title()
            
            y_offset += 16
    
    def _add_svg_excision_guide(self, svg: ET.Element, alignment_info: Dict[str, Any]):
        """Add excision guidance to SVG."""
        
        center = alignment_info['center']
        major_length = alignment_info['major_axis_length']
        minor_length = alignment_info['minor_axis_length']
        angle = alignment_info['rstl_alignment_angle']
        
        # Excision ellipse
        ellipse = ET.SubElement(svg, 'ellipse')
        ellipse.set('cx', f'{center[0]:.1f}')
        ellipse.set('cy', f'{center[1]:.1f}')
        ellipse.set('rx', f'{major_length / 2:.1f}')
        ellipse.set('ry', f'{minor_length / 2:.1f}')
        ellipse.set('transform', f'rotate({angle} {center[0]:.1f} {center[1]:.1f})')
        ellipse.set('class', 'excision-guide')
        ellipse.set('stroke', '#FF0000')
        ellipse.set('stroke-width', '3')
        ellipse.set('fill', 'none')
        ellipse.set('stroke-opacity', '0.8')
        
        # Center point
        center_circle = ET.SubElement(svg, 'circle')
        center_circle.set('cx', f'{center[0]:.1f}')
        center_circle.set('cy', f'{center[1]:.1f}')
        center_circle.set('r', '4')
        center_circle.set('fill', '#FF0000')
        
    def _add_region_labels(self, svg: ET.Element, rstl_lines: List[RSTLLine]):
        """Add region labels to SVG."""
        
        # Calculate center points for each region
        region_centers = {}
        region_counts = {}
        
        for line in rstl_lines:
            if line.confidence < 0.5:
                continue
                
            region = line.region
            center_x = (line.start_point[0] + line.end_point[0]) / 2
            center_y = (line.start_point[1] + line.end_point[1]) / 2
            
            if region not in region_centers:
                region_centers[region] = [0, 0]
                region_counts[region] = 0
            
            region_centers[region][0] += center_x
            region_centers[region][1] += center_y
            region_counts[region] += 1
        
        # Add labels at region centers
        for region, center_sum in region_centers.items():
            count = region_counts[region]
            center_x = center_sum[0] / count
            center_y = center_sum[1] / count
            
            label = ET.SubElement(svg, 'text')
            label.set('x', f'{center_x:.1f}')
            label.set('y', f'{center_y:.1f}')
            label.set('class', 'region-label')
            label.text = region.replace('_', ' ').title()
    
    def _prettify_svg(self, svg_element: ET.Element) -> str:
        """Pretty print SVG element."""
        rough_string = ET.tostring(svg_element, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent='  ')