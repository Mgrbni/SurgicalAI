"""Advanced flap planning with geometric calculations and measurements."""

from typing import Dict, List, Tuple, Optional, NamedTuple
import numpy as np
import math
from dataclasses import dataclass
from enum import Enum

class FlapType(Enum):
    """Enumeration of flap types."""
    ROTATION = "rotation"
    ADVANCEMENT = "advancement"
    TRANSPOSITION = "transposition"
    BILOBED = "bilobed"
    RHOMBOID = "rhomboid"
    V_Y_ADVANCEMENT = "v_y_advancement"
    Z_PLASTY = "z_plasty"

@dataclass
class GeometricPoint:
    """Represents a point in 2D space."""
    x: float
    y: float
    
    def distance_to(self, other: 'GeometricPoint') -> float:
        """Calculate distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_to(self, other: 'GeometricPoint') -> float:
        """Calculate angle to another point in radians."""
        return math.atan2(other.y - self.y, other.x - self.x)

@dataclass
class FlapGeometry:
    """Container for flap geometric parameters."""
    flap_type: FlapType
    pivot_point: GeometricPoint
    primary_arc_angle: float  # in degrees
    secondary_arc_angle: Optional[float]  # for bilobed flaps
    flap_length: float
    flap_width: float
    rotation_angle: float
    tension_vector: Tuple[float, float]
    key_points: List[GeometricPoint]
    estimated_success_rate: float

@dataclass
class DefectAnalysis:
    """Analysis of the defect to be reconstructed."""
    center: GeometricPoint
    major_axis_mm: float
    minor_axis_mm: float
    area_mm2: float
    orientation_angle: float  # angle of major axis
    depth: str  # "partial_thickness", "full_thickness", "deep"
    location: str
    nearby_structures: List[str]

@dataclass
class FlapPlan:
    """Complete flap reconstruction plan."""
    defect_analysis: DefectAnalysis
    recommended_flap: FlapGeometry
    alternative_flaps: List[FlapGeometry]
    warnings: List[str]
    technical_notes: List[str]
    expected_complications: List[str]
    contraindications: List[str]

class AdvancedFlapPlanner:
    """Advanced flap planning with detailed geometric calculations."""
    
    def __init__(self):
        self.golden_ratio = 1.618
        self.max_rotation_angle = 60  # degrees
        self.max_advancement_ratio = 0.7  # maximum advancement as ratio of defect size
        
    def analyze_defect(self, center: Tuple[float, float], 
                      dimensions: Tuple[float, float],
                      location: str,
                      depth: str = "full_thickness",
                      orientation: float = 0.0) -> DefectAnalysis:
        """Analyze defect characteristics for reconstruction planning."""
        
        major_axis = max(dimensions)
        minor_axis = min(dimensions)
        area = math.pi * (major_axis / 2) * (minor_axis / 2)
        
        # Identify nearby critical structures
        nearby_structures = self._identify_nearby_structures(center, location)
        
        return DefectAnalysis(
            center=GeometricPoint(center[0], center[1]),
            major_axis_mm=major_axis,
            minor_axis_mm=minor_axis,
            area_mm2=area,
            orientation_angle=orientation,
            depth=depth,
            location=location,
            nearby_structures=nearby_structures
        )
    
    def plan_reconstruction(self, defect: DefectAnalysis,
                          tissue_laxity: str = "moderate",
                          patient_age: Optional[int] = None,
                          smoking_history: bool = False) -> FlapPlan:
        """Generate comprehensive reconstruction plan."""
        
        # Generate multiple flap options
        flap_options = []
        
        # Primary closure assessment
        if self._assess_primary_closure(defect, tissue_laxity):
            primary_closure = self._plan_primary_closure(defect)
            if primary_closure:
                flap_options.append(primary_closure)
        
        # Local flap options
        rotation_flap = self._plan_rotation_flap(defect, tissue_laxity)
        if rotation_flap:
            flap_options.append(rotation_flap)
        
        advancement_flap = self._plan_advancement_flap(defect, tissue_laxity)
        if advancement_flap:
            flap_options.append(advancement_flap)
        
        transposition_flap = self._plan_transposition_flap(defect, tissue_laxity)
        if transposition_flap:
            flap_options.append(transposition_flap)
        
        # Specialized flaps based on location
        specialized_flaps = self._plan_specialized_flaps(defect)
        flap_options.extend(specialized_flaps)
        
        # Rank flaps by suitability
        ranked_flaps = self._rank_flaps(flap_options, defect, patient_age, smoking_history)
        
        # Generate warnings and contraindications
        warnings = self._generate_warnings(defect, ranked_flaps[0] if ranked_flaps else None)
        contraindications = self._check_contraindications(defect, tissue_laxity, patient_age, smoking_history)
        technical_notes = self._generate_technical_notes(defect, ranked_flaps[0] if ranked_flaps else None)
        expected_complications = self._assess_complications(defect, ranked_flaps[0] if ranked_flaps else None)
        
        return FlapPlan(
            defect_analysis=defect,
            recommended_flap=ranked_flaps[0] if ranked_flaps else None,
            alternative_flaps=ranked_flaps[1:] if len(ranked_flaps) > 1 else [],
            warnings=warnings,
            technical_notes=technical_notes,
            expected_complications=expected_complications,
            contraindications=contraindications
        )
    
    def _assess_primary_closure(self, defect: DefectAnalysis, tissue_laxity: str) -> bool:
        """Assess feasibility of primary closure."""
        
        # Base threshold on tissue laxity
        laxity_thresholds = {
            "high": 25.0,     # mm
            "moderate": 15.0,
            "low": 8.0,
            "poor": 5.0
        }
        
        threshold = laxity_thresholds.get(tissue_laxity, 10.0)
        
        # Adjust for location
        location_factors = {
            "forehead": 1.2,
            "cheek": 1.1,
            "eyelid": 0.6,
            "lip": 0.5,
            "nose": 0.7,
            "ear": 0.8
        }
        
        factor = 1.0
        for loc, mult in location_factors.items():
            if loc in defect.location.lower():
                factor = mult
                break
        
        adjusted_threshold = threshold * factor
        
        return defect.major_axis_mm <= adjusted_threshold
    
    def _plan_primary_closure(self, defect: DefectAnalysis) -> Optional[FlapGeometry]:
        """Plan primary closure if feasible."""
        
        # Calculate optimal closure angle (perpendicular to RSTL if known)
        closure_angle = defect.orientation_angle + 90  # degrees
        
        # Simple linear closure
        key_points = [
            GeometricPoint(defect.center.x - defect.major_axis_mm/2, defect.center.y),
            GeometricPoint(defect.center.x + defect.major_axis_mm/2, defect.center.y)
        ]
        
        return FlapGeometry(
            flap_type=FlapType.ADVANCEMENT,  # Primary closure is technically an advancement
            pivot_point=defect.center,
            primary_arc_angle=0,
            secondary_arc_angle=None,
            flap_length=defect.major_axis_mm,
            flap_width=defect.minor_axis_mm,
            rotation_angle=0,
            tension_vector=(0, 0),
            key_points=key_points,
            estimated_success_rate=0.95
        )
    
    def _plan_rotation_flap(self, defect: DefectAnalysis, tissue_laxity: str) -> Optional[FlapGeometry]:
        """Plan rotation flap reconstruction."""
        
        # Calculate required flap length (typically 3-4x defect diameter)
        flap_length = defect.major_axis_mm * 3.5
        
        # Calculate optimal pivot point (adjacent to defect)
        pivot_distance = defect.major_axis_mm * 0.5
        pivot_angle = defect.orientation_angle + 135  # degrees
        
        pivot_x = defect.center.x + pivot_distance * math.cos(math.radians(pivot_angle))
        pivot_y = defect.center.y + pivot_distance * math.sin(math.radians(pivot_angle))
        pivot_point = GeometricPoint(pivot_x, pivot_y)
        
        # Calculate arc angle (typically 60-90 degrees)
        arc_angle = min(self.max_rotation_angle, 
                       self._calculate_required_rotation(defect, pivot_point))
        
        # Calculate flap width
        flap_width = defect.major_axis_mm * 1.2
        
        # Key points for flap outline
        key_points = self._generate_rotation_flap_points(pivot_point, arc_angle, flap_length, flap_width)
        
        # Estimate success rate based on flap parameters
        success_rate = self._estimate_rotation_success(arc_angle, flap_length, tissue_laxity)
        
        return FlapGeometry(
            flap_type=FlapType.ROTATION,
            pivot_point=pivot_point,
            primary_arc_angle=arc_angle,
            secondary_arc_angle=None,
            flap_length=flap_length,
            flap_width=flap_width,
            rotation_angle=arc_angle,
            tension_vector=self._calculate_tension_vector(pivot_point, defect.center),
            key_points=key_points,
            estimated_success_rate=success_rate
        )
    
    def _plan_advancement_flap(self, defect: DefectAnalysis, tissue_laxity: str) -> Optional[FlapGeometry]:
        """Plan advancement flap reconstruction."""
        
        # Calculate advancement distance
        advancement_distance = defect.major_axis_mm
        
        # Calculate flap dimensions
        flap_length = advancement_distance * 2.5  # Length should be 2.5-3x advancement
        flap_width = defect.major_axis_mm * 1.3
        
        # Position flap adjacent to defect
        flap_angle = defect.orientation_angle + 180  # Opposite to defect orientation
        
        flap_center_x = defect.center.x + (flap_length/2 + defect.major_axis_mm/2) * math.cos(math.radians(flap_angle))
        flap_center_y = defect.center.y + (flap_length/2 + defect.major_axis_mm/2) * math.sin(math.radians(flap_angle))
        
        pivot_point = GeometricPoint(flap_center_x, flap_center_y)
        
        # Generate key points
        key_points = self._generate_advancement_flap_points(pivot_point, flap_length, flap_width, flap_angle)
        
        # Estimate success rate
        success_rate = self._estimate_advancement_success(advancement_distance, flap_length, tissue_laxity)
        
        return FlapGeometry(
            flap_type=FlapType.ADVANCEMENT,
            pivot_point=pivot_point,
            primary_arc_angle=0,
            secondary_arc_angle=None,
            flap_length=flap_length,
            flap_width=flap_width,
            rotation_angle=0,
            tension_vector=(math.cos(math.radians(flap_angle)), math.sin(math.radians(flap_angle))),
            key_points=key_points,
            estimated_success_rate=success_rate
        )
    
    def _plan_transposition_flap(self, defect: DefectAnalysis, tissue_laxity: str) -> Optional[FlapGeometry]:
        """Plan transposition flap reconstruction."""
        
        # Rhomboid/Limberg flap for roughly square/rhomboid defects
        if abs(defect.major_axis_mm - defect.minor_axis_mm) < defect.major_axis_mm * 0.3:
            return self._plan_rhomboid_flap(defect, tissue_laxity)
        
        # General transposition flap
        flap_length = defect.major_axis_mm * 1.5
        flap_width = defect.major_axis_mm * 1.2
        
        # Position at 60-90 degree angle to defect
        transposition_angle = defect.orientation_angle + 75
        
        pivot_distance = defect.major_axis_mm * 0.8
        pivot_x = defect.center.x + pivot_distance * math.cos(math.radians(transposition_angle))
        pivot_y = defect.center.y + pivot_distance * math.sin(math.radians(transposition_angle))
        pivot_point = GeometricPoint(pivot_x, pivot_y)
        
        key_points = self._generate_transposition_flap_points(pivot_point, flap_length, flap_width, transposition_angle)
        
        success_rate = self._estimate_transposition_success(defect, flap_length, tissue_laxity)
        
        return FlapGeometry(
            flap_type=FlapType.TRANSPOSITION,
            pivot_point=pivot_point,
            primary_arc_angle=75,
            secondary_arc_angle=None,
            flap_length=flap_length,
            flap_width=flap_width,
            rotation_angle=75,
            tension_vector=self._calculate_tension_vector(pivot_point, defect.center),
            key_points=key_points,
            estimated_success_rate=success_rate
        )
    
    def _plan_rhomboid_flap(self, defect: DefectAnalysis, tissue_laxity: str) -> FlapGeometry:
        """Plan Limberg/rhomboid flap for square-ish defects."""
        
        # Rhomboid flap dimensions are precisely calculated
        side_length = defect.major_axis_mm * 1.1  # Slightly larger than defect
        
        # Pivot point at one corner of the rhomboid
        pivot_angle = defect.orientation_angle + 120  # 120 degrees from defect center
        pivot_distance = side_length * math.sqrt(3) / 2
        
        pivot_x = defect.center.x + pivot_distance * math.cos(math.radians(pivot_angle))
        pivot_y = defect.center.y + pivot_distance * math.sin(math.radians(pivot_angle))
        pivot_point = GeometricPoint(pivot_x, pivot_y)
        
        # Rhomboid flap has 60-degree angles
        key_points = self._generate_rhomboid_flap_points(pivot_point, side_length)
        
        success_rate = 0.85  # Rhomboid flaps generally have good success rates
        
        return FlapGeometry(
            flap_type=FlapType.RHOMBOID,
            pivot_point=pivot_point,
            primary_arc_angle=60,
            secondary_arc_angle=None,
            flap_length=side_length,
            flap_width=side_length,
            rotation_angle=60,
            tension_vector=self._calculate_tension_vector(pivot_point, defect.center),
            key_points=key_points,
            estimated_success_rate=success_rate
        )
    
    def _plan_specialized_flaps(self, defect: DefectAnalysis) -> List[FlapGeometry]:
        """Plan specialized flaps based on anatomical location."""
        
        specialized_flaps = []
        location = defect.location.lower()
        
        # Nasal tip - bilobed flap
        if "nasal_tip" in location or "nose_tip" in location:
            bilobed = self._plan_bilobed_flap(defect)
            if bilobed:
                specialized_flaps.append(bilobed)
        
        # Forehead - V-Y advancement
        if "forehead" in location:
            vy_flap = self._plan_vy_advancement(defect)
            if vy_flap:
                specialized_flaps.append(vy_flap)
        
        return specialized_flaps
    
    def _plan_bilobed_flap(self, defect: DefectAnalysis) -> Optional[FlapGeometry]:
        """Plan bilobed flap for nasal tip defects."""
        
        # First lobe should be ~50-60% of defect size
        first_lobe_diameter = defect.major_axis_mm * 0.55
        
        # Second lobe should be ~40-50% of defect size  
        second_lobe_diameter = defect.major_axis_mm * 0.45
        
        # Total arc should be ~90-110 degrees
        total_arc = 100
        first_arc = 45
        second_arc = 55
        
        # Pivot point typically at glabella/upper nasal dorsum
        pivot_distance = defect.major_axis_mm * 2.0
        pivot_angle = defect.orientation_angle - 90  # Superior to defect
        
        pivot_x = defect.center.x + pivot_distance * math.cos(math.radians(pivot_angle))
        pivot_y = defect.center.y + pivot_distance * math.sin(math.radians(pivot_angle))
        pivot_point = GeometricPoint(pivot_x, pivot_y)
        
        # Generate bilobed flap outline
        key_points = self._generate_bilobed_flap_points(
            pivot_point, first_lobe_diameter, second_lobe_diameter, first_arc, second_arc
        )
        
        return FlapGeometry(
            flap_type=FlapType.BILOBED,
            pivot_point=pivot_point,
            primary_arc_angle=first_arc,
            secondary_arc_angle=second_arc,
            flap_length=first_lobe_diameter,
            flap_width=first_lobe_diameter,
            rotation_angle=total_arc,
            tension_vector=self._calculate_tension_vector(pivot_point, defect.center),
            key_points=key_points,
            estimated_success_rate=0.80
        )
    
    def _plan_vy_advancement(self, defect: DefectAnalysis) -> Optional[FlapGeometry]:
        """Plan V-Y advancement flap."""
        
        # V-Y flap dimensions
        flap_length = defect.major_axis_mm * 2.0
        flap_width = defect.major_axis_mm * 1.2
        
        # V-shaped flap with apex pointing away from defect
        apex_distance = flap_length
        apex_angle = defect.orientation_angle + 180
        
        apex_x = defect.center.x + apex_distance * math.cos(math.radians(apex_angle))
        apex_y = defect.center.y + apex_distance * math.sin(math.radians(apex_angle))
        apex_point = GeometricPoint(apex_x, apex_y)
        
        key_points = self._generate_vy_flap_points(defect.center, apex_point, flap_width)
        
        return FlapGeometry(
            flap_type=FlapType.V_Y_ADVANCEMENT,
            pivot_point=apex_point,
            primary_arc_angle=0,
            secondary_arc_angle=None,
            flap_length=flap_length,
            flap_width=flap_width,
            rotation_angle=0,
            tension_vector=(0, 0),
            key_points=key_points,
            estimated_success_rate=0.85
        )
    
    def _identify_nearby_structures(self, center: Tuple[float, float], location: str) -> List[str]:
        """Identify critical anatomical structures near the defect."""
        
        structures = []
        location_lower = location.lower()
        
        if "eyelid" in location_lower:
            structures.extend(["lacrimal_system", "levator_muscle", "tarsal_plate"])
        if "nose" in location_lower:
            structures.extend(["nasal_cartilage", "nasal_septum"])
        if "lip" in location_lower:
            structures.extend(["vermillion_border", "orbicularis_oris"])
        if "ear" in location_lower:
            structures.extend(["auricular_cartilage", "external_auditory_canal"])
        if "forehead" in location_lower:
            structures.extend(["supraorbital_nerve", "frontalis_muscle"])
        if "cheek" in location_lower:
            structures.extend(["facial_nerve", "parotid_duct"])
        
        return structures
    
    def _rank_flaps(self, flap_options: List[FlapGeometry], defect: DefectAnalysis,
                   patient_age: Optional[int], smoking_history: bool) -> List[FlapGeometry]:
        """Rank flap options by suitability."""
        
        def calculate_score(flap: FlapGeometry) -> float:
            score = flap.estimated_success_rate
            
            # Prefer simpler flaps
            complexity_penalty = {
                FlapType.ADVANCEMENT: 0,
                FlapType.ROTATION: 0.05,
                FlapType.TRANSPOSITION: 0.1,
                FlapType.BILOBED: 0.15,
                FlapType.RHOMBOID: 0.1,
                FlapType.V_Y_ADVANCEMENT: 0.08,
                FlapType.Z_PLASTY: 0.12
            }
            score -= complexity_penalty.get(flap.flap_type, 0.1)
            
            # Age factor (elderly patients - prefer simpler procedures)
            if patient_age and patient_age > 70:
                if flap.flap_type in [FlapType.BILOBED, FlapType.Z_PLASTY]:
                    score -= 0.1
            
            # Smoking penalty for complex flaps
            if smoking_history:
                if flap.rotation_angle > 45 or flap.flap_length > defect.major_axis_mm * 3:
                    score -= 0.15
            
            return score
        
        return sorted(flap_options, key=calculate_score, reverse=True)
    
    def _generate_warnings(self, defect: DefectAnalysis, recommended_flap: Optional[FlapGeometry]) -> List[str]:
        """Generate warnings for the reconstruction plan."""
        
        warnings = []
        
        if defect.major_axis_mm > 30:
            warnings.append("Large defect size may require staged reconstruction")
        
        if "eyelid" in defect.location.lower():
            warnings.append("Risk of ectropion - ensure adequate lid support")
        
        if "lip" in defect.location.lower():
            warnings.append("Preserve vermillion border alignment")
        
        if recommended_flap and recommended_flap.rotation_angle > 60:
            warnings.append("Large rotation angle increases risk of flap necrosis")
        
        if defect.area_mm2 > 500:
            warnings.append("Consider free tissue transfer for large defects")
        
        return warnings
    
    def _check_contraindications(self, defect: DefectAnalysis, tissue_laxity: str,
                               patient_age: Optional[int], smoking_history: bool) -> List[str]:
        """Check for contraindications to local flap reconstruction."""
        
        contraindications = []
        
        if smoking_history and defect.major_axis_mm > 20:
            contraindications.append("Active smoking increases flap necrosis risk")
        
        if tissue_laxity == "poor" and defect.major_axis_mm > 15:
            contraindications.append("Poor tissue laxity limits local flap options")
        
        if patient_age and patient_age > 80 and defect.area_mm2 > 300:
            contraindications.append("Consider simpler reconstruction in elderly patients")
        
        return contraindications
    
    def _generate_technical_notes(self, defect: DefectAnalysis, recommended_flap: Optional[FlapGeometry]) -> List[str]:
        """Generate technical surgical notes."""
        
        notes = []
        
        if recommended_flap:
            if recommended_flap.flap_type == FlapType.ROTATION:
                notes.append(f"Back-cut angle: {recommended_flap.primary_arc_angle}°")
                notes.append("Undermine in subcutaneous plane")
                
            elif recommended_flap.flap_type == FlapType.BILOBED:
                notes.append("First lobe covers primary defect")
                notes.append("Second lobe covers first lobe donor site")
                notes.append("Preserve vascular pedicle")
                
            elif recommended_flap.flap_type == FlapType.ADVANCEMENT:
                notes.append("Score deep dermis for advancement")
                notes.append("Consider Burow's triangles for dog-ear prevention")
        
        notes.append("Layer closure essential for optimal healing")
        notes.append("Consider postoperative splinting if needed")
        
        return notes
    
    def _assess_complications(self, defect: DefectAnalysis, recommended_flap: Optional[FlapGeometry]) -> List[str]:
        """Assess potential complications."""
        
        complications = ["bleeding", "infection", "dehiscence"]
        
        if recommended_flap:
            if recommended_flap.flap_length > defect.major_axis_mm * 3:
                complications.append("flap tip necrosis")
            
            if recommended_flap.rotation_angle > 45:
                complications.append("wound tension")
                
            if defect.location.lower() in ["eyelid", "lip"]:
                complications.append("functional impairment")
        
        return complications
    
    # Utility methods for geometric calculations
    def _calculate_required_rotation(self, defect: DefectAnalysis, pivot: GeometricPoint) -> float:
        """Calculate required rotation angle for flap."""
        distance = pivot.distance_to(defect.center)
        return min(math.degrees(defect.major_axis_mm / distance), self.max_rotation_angle)
    
    def _calculate_tension_vector(self, pivot: GeometricPoint, target: GeometricPoint) -> Tuple[float, float]:
        """Calculate tension vector direction."""
        angle = pivot.angle_to(target)
        return (math.cos(angle), math.sin(angle))
    
    def _estimate_rotation_success(self, arc_angle: float, flap_length: float, tissue_laxity: str) -> float:
        """Estimate success rate for rotation flap."""
        base_rate = 0.85
        
        # Penalize large rotation angles
        if arc_angle > 45:
            base_rate -= (arc_angle - 45) * 0.01
        
        # Adjust for tissue laxity
        laxity_bonus = {"high": 0.1, "moderate": 0.05, "low": -0.05, "poor": -0.15}.get(tissue_laxity, 0)
        
        return max(0.5, base_rate + laxity_bonus)
    
    def _estimate_advancement_success(self, advancement: float, flap_length: float, tissue_laxity: str) -> float:
        """Estimate success rate for advancement flap."""
        base_rate = 0.90
        
        # Check advancement ratio
        advancement_ratio = advancement / flap_length
        if advancement_ratio > self.max_advancement_ratio:
            base_rate -= (advancement_ratio - self.max_advancement_ratio) * 0.5
        
        laxity_bonus = {"high": 0.05, "moderate": 0, "low": -0.1, "poor": -0.2}.get(tissue_laxity, 0)
        
        return max(0.6, base_rate + laxity_bonus)
    
    def _estimate_transposition_success(self, defect: DefectAnalysis, flap_length: float, tissue_laxity: str) -> float:
        """Estimate success rate for transposition flap."""
        base_rate = 0.80
        
        # Adjust for defect size relative to flap
        size_ratio = defect.major_axis_mm / flap_length
        if size_ratio > 0.8:
            base_rate -= 0.1
        
        laxity_bonus = {"high": 0.1, "moderate": 0.05, "low": -0.05, "poor": -0.15}.get(tissue_laxity, 0)
        
        return max(0.5, base_rate + laxity_bonus)
    
    # Geometric point generation methods (simplified implementations)
    def _generate_rotation_flap_points(self, pivot: GeometricPoint, arc_angle: float, 
                                     length: float, width: float) -> List[GeometricPoint]:
        """Generate key points for rotation flap outline."""
        points = []
        # Simplified - in practice would generate full flap boundary
        points.append(pivot)
        points.append(GeometricPoint(pivot.x + length, pivot.y))
        points.append(GeometricPoint(pivot.x + length * math.cos(math.radians(arc_angle)), 
                                   pivot.y + length * math.sin(math.radians(arc_angle))))
        return points
    
    def _generate_advancement_flap_points(self, center: GeometricPoint, length: float, 
                                        width: float, angle: float) -> List[GeometricPoint]:
        """Generate key points for advancement flap outline."""
        points = []
        # Simplified rectangular flap
        half_width = width / 2
        half_length = length / 2
        
        corners = [(-half_length, -half_width), (half_length, -half_width), 
                  (half_length, half_width), (-half_length, half_width)]
        
        for dx, dy in corners:
            x = center.x + dx * math.cos(math.radians(angle)) - dy * math.sin(math.radians(angle))
            y = center.y + dx * math.sin(math.radians(angle)) + dy * math.cos(math.radians(angle))
            points.append(GeometricPoint(x, y))
        
        return points
    
    def _generate_transposition_flap_points(self, pivot: GeometricPoint, length: float, 
                                          width: float, angle: float) -> List[GeometricPoint]:
        """Generate key points for transposition flap outline."""
        # Similar to advancement but with pivot-based geometry
        return self._generate_advancement_flap_points(pivot, length, width, angle)
    
    def _generate_rhomboid_flap_points(self, pivot: GeometricPoint, side_length: float) -> List[GeometricPoint]:
        """Generate key points for rhomboid flap outline."""
        points = []
        # Rhomboid with 60/120 degree angles
        angles = [0, 60, 120, 180]
        for angle in angles:
            x = pivot.x + side_length * math.cos(math.radians(angle))
            y = pivot.y + side_length * math.sin(math.radians(angle))
            points.append(GeometricPoint(x, y))
        return points
    
    def _generate_bilobed_flap_points(self, pivot: GeometricPoint, first_diameter: float, 
                                    second_diameter: float, first_arc: float, second_arc: float) -> List[GeometricPoint]:
        """Generate key points for bilobed flap outline."""
        points = []
        # First lobe
        points.append(GeometricPoint(pivot.x + first_diameter, pivot.y))
        points.append(GeometricPoint(
            pivot.x + first_diameter * math.cos(math.radians(first_arc)),
            pivot.y + first_diameter * math.sin(math.radians(first_arc))
        ))
        
        # Second lobe
        second_start_angle = first_arc + 10  # Small gap between lobes
        points.append(GeometricPoint(
            pivot.x + second_diameter * math.cos(math.radians(second_start_angle)),
            pivot.y + second_diameter * math.sin(math.radians(second_start_angle))
        ))
        points.append(GeometricPoint(
            pivot.x + second_diameter * math.cos(math.radians(second_start_angle + second_arc)),
            pivot.y + second_diameter * math.sin(math.radians(second_start_angle + second_arc))
        ))
        
        return points
    
    def _generate_vy_flap_points(self, base_center: GeometricPoint, apex: GeometricPoint, 
                               width: float) -> List[GeometricPoint]:
        """Generate key points for V-Y advancement flap outline."""
        points = []
        
        # Calculate perpendicular direction for base width
        base_to_apex = (apex.x - base_center.x, apex.y - base_center.y)
        length = math.sqrt(base_to_apex[0]**2 + base_to_apex[1]**2)
        perp = (-base_to_apex[1] / length, base_to_apex[0] / length)
        
        # V-shape base points
        half_width = width / 2
        points.append(GeometricPoint(
            base_center.x + perp[0] * half_width,
            base_center.y + perp[1] * half_width
        ))
        points.append(GeometricPoint(
            base_center.x - perp[0] * half_width,
            base_center.y - perp[1] * half_width
        ))
        
        # Apex
        points.append(apex)
        
        return points