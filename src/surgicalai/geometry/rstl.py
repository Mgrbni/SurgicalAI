"""RSTL (Relaxed Skin Tension Lines) atlas and alignment system."""

from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
import cv2
from dataclasses import dataclass
import json
from pathlib import Path

from ..vision.landmarks import FacialLandmarks, HeadPose

@dataclass
class RSTLLine:
    """Represents a single RSTL line."""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    region: str
    direction_vector: Tuple[float, float]
    confidence: float

@dataclass
class RSTLAtlas:
    """Atlas of RSTL lines for facial regions."""
    lines: List[RSTLLine]
    reference_landmarks: Dict[str, Tuple[float, float]]
    atlas_size: Tuple[int, int]

class RSTLAnalyzer:
    """Analyzes and overlays RSTL information on facial images."""
    
    def __init__(self):
        self.atlas = self._create_rstl_atlas()
    
    def _create_rstl_atlas(self) -> RSTLAtlas:
        """Create a reference RSTL atlas based on facial anatomy."""
        # Reference atlas on a normalized 512x512 face
        atlas_size = (512, 512)
        
        # Reference landmarks for a normalized frontal face
        reference_landmarks = {
            'left_eye': (150, 180),
            'right_eye': (362, 180),
            'nose_tip': (256, 280),
            'mouth_left': (200, 350),
            'mouth_right': (312, 350),
            'chin': (256, 450),
            'forehead': (256, 80),
            'left_cheek': (120, 320),
            'right_cheek': (392, 320),
        }
        
        # Define RSTL lines based on anatomical knowledge
        lines = []
        
        # Forehead - horizontal lines
        for y in range(60, 140, 20):
            lines.append(RSTLLine(
                start_point=(100, y),
                end_point=(412, y),
                region="forehead",
                direction_vector=(1.0, 0.0),
                confidence=0.9
            ))
        
        # Glabellar region - vertical
        lines.append(RSTLLine(
            start_point=(256, 120),
            end_point=(256, 160),
            region="glabella",
            direction_vector=(0.0, 1.0),
            confidence=0.8
        ))
        
        # Periorbital - radial around eyes
        eye_centers = [(150, 180), (362, 180)]
        for i, (ex, ey) in enumerate(eye_centers):
            region = "left_periorbital" if i == 0 else "right_periorbital"
            # Upper lid
            lines.append(RSTLLine(
                start_point=(ex - 30, ey - 15),
                end_point=(ex + 30, ey - 15),
                region=region,
                direction_vector=(1.0, 0.0),
                confidence=0.7
            ))
            # Lower lid
            lines.append(RSTLLine(
                start_point=(ex - 25, ey + 15),
                end_point=(ex + 25, ey + 15),
                region=region,
                direction_vector=(1.0, 0.0),
                confidence=0.7
            ))
        
        # Nasal dorsum - vertical
        lines.append(RSTLLine(
            start_point=(256, 160),
            end_point=(256, 260),
            region="nasal_dorsum",
            direction_vector=(0.0, 1.0),
            confidence=0.9
        ))
        
        # Nasal tip - radial
        nose_center = (256, 280)
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = np.radians(angle)
            dx, dy = np.cos(rad) * 20, np.sin(rad) * 20
            lines.append(RSTLLine(
                start_point=(nose_center[0] - dx, nose_center[1] - dy),
                end_point=(nose_center[0] + dx, nose_center[1] + dy),
                region="nasal_tip",
                direction_vector=(np.cos(rad), np.sin(rad)),
                confidence=0.6
            ))
        
        # Cheeks - curved lines following facial contours
        # Left cheek
        cheek_lines_left = [
            ((80, 250), (180, 280), "left_cheek"),
            ((70, 280), (170, 320), "left_cheek"),
            ((80, 320), (180, 360), "left_cheek"),
        ]
        
        # Right cheek (mirrored)
        cheek_lines_right = [
            ((332, 280), (432, 250), "right_cheek"),
            ((342, 320), (442, 280), "right_cheek"),
            ((332, 360), (432, 320), "right_cheek"),
        ]
        
        for (start, end, region) in cheek_lines_left + cheek_lines_right:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            direction = (dx/length, dy/length) if length > 0 else (1.0, 0.0)
            
            lines.append(RSTLLine(
                start_point=start,
                end_point=end,
                region=region,
                direction_vector=direction,
                confidence=0.8
            ))
        
        # Perioral - radial around mouth
        mouth_center = (256, 350)
        for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            rad = np.radians(angle)
            dx, dy = np.cos(rad) * 25, np.sin(rad) * 15
            lines.append(RSTLLine(
                start_point=(mouth_center[0] - dx, mouth_center[1] - dy),
                end_point=(mouth_center[0] + dx, mouth_center[1] + dy),
                region="perioral",
                direction_vector=(np.cos(rad), np.sin(rad)),
                confidence=0.7
            ))
        
        # Chin - horizontal
        lines.append(RSTLLine(
            start_point=(180, 450),
            end_point=(332, 450),
            region="chin",
            direction_vector=(1.0, 0.0),
            confidence=0.8
        ))
        
        # Mandibular - following jaw line
        jaw_points = [(120, 400), (180, 430), (256, 450), (332, 430), (392, 400)]
        for i in range(len(jaw_points) - 1):
            start, end = jaw_points[i], jaw_points[i + 1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            direction = (dx/length, dy/length) if length > 0 else (1.0, 0.0)
            
            lines.append(RSTLLine(
                start_point=start,
                end_point=end,
                region="mandibular",
                direction_vector=direction,
                confidence=0.8
            ))
        
        return RSTLAtlas(
            lines=lines,
            reference_landmarks=reference_landmarks,
            atlas_size=atlas_size
        )
    
    def align_rstl_to_face(self, landmarks: FacialLandmarks, 
                          image_shape: Tuple[int, int],
                          head_pose: Optional[HeadPose] = None) -> List[RSTLLine]:
        """Align RSTL atlas to detected facial landmarks."""
        
        h, w = image_shape[:2]
        
        # Calculate transformation from atlas to current face
        transform_matrix = self._calculate_transformation(landmarks, image_shape)
        
        # Apply head pose correction if available
        if head_pose:
            pose_correction = self._calculate_pose_correction(head_pose)
            transform_matrix = np.dot(transform_matrix, pose_correction)
        
        # Transform RSTL lines
        aligned_lines = []
        for line in self.atlas.lines:
            # Transform start and end points
            start_transformed = self._transform_point(line.start_point, transform_matrix)
            end_transformed = self._transform_point(line.end_point, transform_matrix)
            
            # Calculate new direction vector
            dx = end_transformed[0] - start_transformed[0]
            dy = end_transformed[1] - start_transformed[1]
            length = np.sqrt(dx**2 + dy**2)
            new_direction = (dx/length, dy/length) if length > 0 else line.direction_vector
            
            # Adjust confidence based on pose and transformation quality
            confidence_factor = self._calculate_confidence_factor(head_pose, landmarks.confidence)
            new_confidence = line.confidence * confidence_factor
            
            aligned_lines.append(RSTLLine(
                start_point=start_transformed,
                end_point=end_transformed,
                region=line.region,
                direction_vector=new_direction,
                confidence=new_confidence
            ))
        
        return aligned_lines
    
    def _calculate_transformation(self, landmarks: FacialLandmarks, 
                                image_shape: Tuple[int, int]) -> np.ndarray:
        """Calculate transformation matrix from atlas to current face."""
        
        # Key landmark correspondences
        atlas_points = np.array([
            self.atlas.reference_landmarks['left_eye'],
            self.atlas.reference_landmarks['right_eye'],
            self.atlas.reference_landmarks['nose_tip'],
            self.atlas.reference_landmarks['chin']
        ], dtype=np.float32)
        
        face_points = np.array([
            landmarks.left_eye,
            landmarks.right_eye,
            landmarks.nose_tip,
            landmarks.chin
        ], dtype=np.float32)
        
        # Calculate similarity transformation (scale, rotation, translation)
        transform_matrix = cv2.estimateAffinePartial2D(atlas_points, face_points)[0]
        
        if transform_matrix is None:
            # Fallback to simple scaling and translation
            atlas_center = np.mean(atlas_points, axis=0)
            face_center = np.mean(face_points, axis=0)
            
            # Calculate scale based on eye distance
            atlas_eye_dist = np.linalg.norm(atlas_points[1] - atlas_points[0])
            face_eye_dist = np.linalg.norm(face_points[1] - face_points[0])
            scale = face_eye_dist / atlas_eye_dist if atlas_eye_dist > 0 else 1.0
            
            # Translation
            translation = face_center - atlas_center * scale
            
            transform_matrix = np.array([
                [scale, 0, translation[0]],
                [0, scale, translation[1]]
            ], dtype=np.float32)
        
        return transform_matrix
    
    def _calculate_pose_correction(self, head_pose: HeadPose) -> np.ndarray:
        """Calculate pose correction matrix for head rotation."""
        
        # Convert pose angles to correction factors
        yaw_rad = np.radians(head_pose.yaw)
        pitch_rad = np.radians(head_pose.pitch)
        roll_rad = np.radians(head_pose.roll)
        
        # Create rotation matrices
        # Yaw (left-right rotation)
        cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
        yaw_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        # Pitch affects vertical compression
        pitch_factor = np.cos(pitch_rad)
        
        # Roll rotation
        cos_roll, sin_roll = np.cos(roll_rad), np.sin(roll_rad)
        roll_matrix = np.array([
            [cos_roll, -sin_roll],
            [sin_roll, cos_roll]
        ])
        
        # Combine transformations
        combined_rotation = np.dot(roll_matrix, yaw_matrix)
        
        # Apply pitch scaling
        pitch_scaling = np.array([
            [1.0, 0.0],
            [0.0, pitch_factor]
        ])
        
        final_correction = np.dot(combined_rotation, pitch_scaling)
        
        # Convert to 2x3 affine matrix
        correction_matrix = np.array([
            [final_correction[0, 0], final_correction[0, 1], 0],
            [final_correction[1, 0], final_correction[1, 1], 0]
        ], dtype=np.float32)
        
        return correction_matrix
    
    def _transform_point(self, point: Tuple[float, float], 
                        transform_matrix: np.ndarray) -> Tuple[float, float]:
        """Transform a point using the transformation matrix."""
        
        point_array = np.array([point[0], point[1], 1.0], dtype=np.float32)
        transformed = np.dot(transform_matrix, point_array)
        return (float(transformed[0]), float(transformed[1]))
    
    def _calculate_confidence_factor(self, head_pose: Optional[HeadPose], 
                                   landmark_confidence: float) -> float:
        """Calculate confidence factor based on pose and landmark quality."""
        
        confidence_factor = landmark_confidence
        
        if head_pose:
            # Reduce confidence for extreme poses
            pose_factor = 1.0
            if abs(head_pose.yaw) > 30:
                pose_factor *= 0.7
            if abs(head_pose.pitch) > 20:
                pose_factor *= 0.8
            if abs(head_pose.roll) > 15:
                pose_factor *= 0.9
            
            confidence_factor *= pose_factor * head_pose.confidence
        
        return confidence_factor
    
    def get_local_rstl_direction(self, point: Tuple[float, float], 
                               aligned_lines: List[RSTLLine],
                               search_radius: float = 50.0) -> Optional[Tuple[float, float]]:
        """Get local RSTL direction at a specific point."""
        
        nearby_lines = []
        
        for line in aligned_lines:
            # Calculate distance from point to line
            distance = self._point_to_line_distance(point, line)
            
            if distance <= search_radius:
                # Weight by confidence and inverse distance
                weight = line.confidence / (1.0 + distance / search_radius)
                nearby_lines.append((line, weight))
        
        if not nearby_lines:
            return None
        
        # Calculate weighted average direction
        total_weight = sum(weight for _, weight in nearby_lines)
        weighted_direction = [0.0, 0.0]
        
        for line, weight in nearby_lines:
            weighted_direction[0] += line.direction_vector[0] * weight
            weighted_direction[1] += line.direction_vector[1] * weight
        
        weighted_direction[0] /= total_weight
        weighted_direction[1] /= total_weight
        
        # Normalize
        length = np.sqrt(weighted_direction[0]**2 + weighted_direction[1]**2)
        if length > 0:
            return (weighted_direction[0] / length, weighted_direction[1] / length)
        
        return None
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                              line: RSTLLine) -> float:
        """Calculate minimum distance from point to line segment."""
        
        px, py = point
        x1, y1 = line.start_point
        x2, y2 = line.end_point
        
        # Vector from line start to point
        dx_p = px - x1
        dy_p = py - y1
        
        # Line vector
        dx_l = x2 - x1
        dy_l = y2 - y1
        
        line_length_sq = dx_l**2 + dy_l**2
        
        if line_length_sq == 0:
            # Line is a point
            return np.sqrt(dx_p**2 + dy_p**2)
        
        # Parameter t for closest point on line
        t = max(0, min(1, (dx_p * dx_l + dy_p * dy_l) / line_length_sq))
        
        # Closest point on line
        closest_x = x1 + t * dx_l
        closest_y = y1 + t * dy_l
        
        # Distance from point to closest point on line
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def calculate_optimal_excision_alignment(self, lesion_center: Tuple[float, float],
                                          aligned_lines: List[RSTLLine],
                                          lesion_size: Tuple[float, float]) -> Dict[str, any]:
        """Calculate optimal excision alignment with RSTL."""
        
        # Get local RSTL direction at lesion center
        rstl_direction = self.get_local_rstl_direction(lesion_center, aligned_lines)
        
        if rstl_direction is None:
            # Fallback to horizontal if no RSTL found
            rstl_direction = (1.0, 0.0)
        
        # Calculate excision long axis aligned with RSTL
        long_axis_direction = rstl_direction
        
        # Calculate perpendicular direction for short axis
        short_axis_direction = (-rstl_direction[1], rstl_direction[0])
        
        # Determine excision dimensions (elliptical with 3:1 ratio typical)
        lesion_width, lesion_height = lesion_size
        major_axis_length = max(lesion_width, lesion_height) * 1.5  # Add margin
        minor_axis_length = major_axis_length / 3.0
        
        return {
            'center': lesion_center,
            'long_axis_direction': long_axis_direction,
            'short_axis_direction': short_axis_direction,
            'major_axis_length': major_axis_length,
            'minor_axis_length': minor_axis_length,
            'rstl_alignment_angle': np.degrees(np.arctan2(rstl_direction[1], rstl_direction[0])),
            'confidence': self._calculate_alignment_confidence(lesion_center, aligned_lines)
        }
    
    def _calculate_alignment_confidence(self, point: Tuple[float, float],
                                      aligned_lines: List[RSTLLine]) -> float:
        """Calculate confidence in RSTL alignment at given point."""
        
        nearby_lines = [line for line in aligned_lines 
                       if self._point_to_line_distance(point, line) <= 50.0]
        
        if not nearby_lines:
            return 0.3  # Low confidence without nearby RSTL
        
        # Average confidence of nearby lines
        avg_confidence = np.mean([line.confidence for line in nearby_lines])
        
        # Bonus for multiple nearby lines (consensus)
        consensus_bonus = min(len(nearby_lines) * 0.1, 0.3)
        
        return min(avg_confidence + consensus_bonus, 1.0)