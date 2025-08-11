"""Facial landmark detection for RSTL alignment and analysis."""

from typing import Dict, Tuple, List, Optional, NamedTuple
import numpy as np
import cv2
from dataclasses import dataclass
import logging

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available, using fallback landmark detection")

@dataclass
class FacialLandmarks:
    """Container for facial landmark points."""
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose_tip: Tuple[float, float]
    mouth_left: Tuple[float, float]
    mouth_right: Tuple[float, float]
    chin: Tuple[float, float]
    forehead: Tuple[float, float]
    left_cheek: Tuple[float, float]
    right_cheek: Tuple[float, float]
    confidence: float
    all_points: Dict[str, Tuple[float, float]]

@dataclass 
class HeadPose:
    """Container for head pose estimation."""
    yaw: float      # Left-right rotation
    pitch: float    # Up-down rotation  
    roll: float     # Tilt rotation
    confidence: float

class FacialLandmarkDetector:
    """Facial landmark detection using MediaPipe or fallback methods."""
    
    def __init__(self):
        self.mp_detector = None
        if MEDIAPIPE_AVAILABLE:
            self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe face mesh detector."""
        try:
            mp_face_mesh = mp.solutions.face_mesh
            self.mp_detector = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except Exception as e:
            logging.warning(f"MediaPipe initialization failed: {e}")
            self.mp_detector = None
    
    def detect_landmarks(self, rgb_image: np.ndarray) -> Optional[FacialLandmarks]:
        """Detect facial landmarks in RGB image."""
        
        if self.mp_detector is not None:
            return self._detect_mediapipe(rgb_image)
        else:
            return self._detect_fallback(rgb_image)
    
    def _detect_mediapipe(self, rgb_image: np.ndarray) -> Optional[FacialLandmarks]:
        """Detect landmarks using MediaPipe."""
        try:
            results = self.mp_detector.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            face_landmarks = results.multi_face_landmarks[0]
            h, w = rgb_image.shape[:2]
            
            # Extract key landmarks using MediaPipe indices
            landmarks_dict = {}
            all_points = {}
            
            # MediaPipe face mesh landmark indices for key features
            key_indices = {
                'left_eye': 159,       # Left eye center
                'right_eye': 386,      # Right eye center  
                'nose_tip': 1,         # Nose tip
                'mouth_left': 61,      # Left mouth corner
                'mouth_right': 291,    # Right mouth corner
                'chin': 175,           # Chin center
                'forehead': 9,         # Forehead center
                'left_cheek': 116,     # Left cheek
                'right_cheek': 345,    # Right cheek
            }
            
            confidence = 0.8  # MediaPipe generally has good confidence
            
            for name, idx in key_indices.items():
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    x = landmark.x * w
                    y = landmark.y * h
                    landmarks_dict[name] = (x, y)
                    all_points[name] = (x, y)
            
            # Store all landmarks for potential future use
            for i, landmark in enumerate(face_landmarks.landmark):
                all_points[f'point_{i}'] = (landmark.x * w, landmark.y * h)
            
            return FacialLandmarks(
                left_eye=landmarks_dict.get('left_eye', (0, 0)),
                right_eye=landmarks_dict.get('right_eye', (0, 0)),
                nose_tip=landmarks_dict.get('nose_tip', (0, 0)),
                mouth_left=landmarks_dict.get('mouth_left', (0, 0)),
                mouth_right=landmarks_dict.get('mouth_right', (0, 0)),
                chin=landmarks_dict.get('chin', (0, 0)),
                forehead=landmarks_dict.get('forehead', (0, 0)),
                left_cheek=landmarks_dict.get('left_cheek', (0, 0)),
                right_cheek=landmarks_dict.get('right_cheek', (0, 0)),
                confidence=confidence,
                all_points=all_points
            )
            
        except Exception as e:
            logging.warning(f"MediaPipe landmark detection failed: {e}")
            return self._detect_fallback(rgb_image)
    
    def _detect_fallback(self, rgb_image: np.ndarray) -> Optional[FacialLandmarks]:
        """Fallback landmark detection using OpenCV Haar cascades."""
        try:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            # Try to detect face first
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, fw, fh = face
            
            # Estimate landmark positions based on face bounding box
            # These are rough estimates based on typical face proportions
            landmarks_dict = {
                'left_eye': (x + fw * 0.3, y + fh * 0.35),
                'right_eye': (x + fw * 0.7, y + fh * 0.35),
                'nose_tip': (x + fw * 0.5, y + fh * 0.55),
                'mouth_left': (x + fw * 0.35, y + fh * 0.75),
                'mouth_right': (x + fw * 0.65, y + fh * 0.75),
                'chin': (x + fw * 0.5, y + fh * 0.95),
                'forehead': (x + fw * 0.5, y + fh * 0.15),
                'left_cheek': (x + fw * 0.25, y + fh * 0.6),
                'right_cheek': (x + fw * 0.75, y + fh * 0.6),
            }
            
            return FacialLandmarks(
                left_eye=landmarks_dict['left_eye'],
                right_eye=landmarks_dict['right_eye'],
                nose_tip=landmarks_dict['nose_tip'],
                mouth_left=landmarks_dict['mouth_left'],
                mouth_right=landmarks_dict['mouth_right'],
                chin=landmarks_dict['chin'],
                forehead=landmarks_dict['forehead'],
                left_cheek=landmarks_dict['left_cheek'],
                right_cheek=landmarks_dict['right_cheek'],
                confidence=0.5,  # Lower confidence for fallback method
                all_points=landmarks_dict
            )
            
        except Exception as e:
            logging.warning(f"Fallback landmark detection failed: {e}")
            return None
    
    def estimate_head_pose(self, landmarks: FacialLandmarks, image_shape: Tuple[int, int]) -> HeadPose:
        """Estimate head pose from facial landmarks."""
        h, w = image_shape[:2]
        
        # 3D model points (approximate)
        model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -330.0, -65.0),   # Chin
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])
        
        # 2D image points
        image_points = np.array([
            landmarks.nose_tip,
            landmarks.chin,
            landmarks.left_eye,
            landmarks.right_eye,
            landmarks.mouth_left,
            landmarks.mouth_right
        ], dtype="double")
        
        # Camera internals (estimated)
        focal_length = w
        center = (w // 2, h // 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        # Distortion coefficients (assuming no distortion)
        dist_coeffs = np.zeros((4, 1))
        
        try:
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
                
                return HeadPose(
                    yaw=np.degrees(yaw),
                    pitch=np.degrees(pitch),
                    roll=np.degrees(roll),
                    confidence=landmarks.confidence
                )
        except Exception as e:
            logging.warning(f"Head pose estimation failed: {e}")
        
        # Fallback: estimate pose from landmark positions
        return self._estimate_pose_fallback(landmarks, image_shape)
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles."""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return x, y, z
    
    def _estimate_pose_fallback(self, landmarks: FacialLandmarks, image_shape: Tuple[int, int]) -> HeadPose:
        """Fallback head pose estimation using simple geometric relationships."""
        h, w = image_shape[:2]
        
        # Calculate face center
        face_center_x = (landmarks.left_eye[0] + landmarks.right_eye[0]) / 2
        face_center_y = (landmarks.left_eye[1] + landmarks.right_eye[1]) / 2
        
        # Estimate yaw from horizontal position
        yaw = (face_center_x - w/2) / (w/2) * 30  # Scale to ±30 degrees
        
        # Estimate pitch from vertical position
        pitch = (face_center_y - h/2) / (h/2) * 20  # Scale to ±20 degrees
        
        # Estimate roll from eye line angle
        eye_line_angle = np.arctan2(
            landmarks.right_eye[1] - landmarks.left_eye[1],
            landmarks.right_eye[0] - landmarks.left_eye[0]
        )
        roll = np.degrees(eye_line_angle)
        
        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            confidence=landmarks.confidence * 0.7  # Lower confidence for fallback
        )