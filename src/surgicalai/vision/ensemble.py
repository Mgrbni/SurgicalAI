"""Ensemble classifier system for lesion diagnosis."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class DiagnosisResult:
    """Container for diagnosis results with confidence metrics."""
    diagnosis: str
    probability: float
    confidence_interval: Tuple[float, float]
    confidence_score: float

@dataclass
class EnsembleOutput:
    """Container for ensemble classifier output."""
    top_diagnoses: List[DiagnosisResult]
    uncertainty_score: float
    high_risk_flag: bool
    feature_explanations: Dict[str, float]

class LesionClassifier:
    """Individual classifier for lesion analysis."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict probabilities for each diagnosis class."""
        # Mock implementation - in real use, this would load and run actual models
        # For demo purposes, we'll use feature-based heuristics
        
        if self.name == "abcde_classifier":
            return self._abcde_based_prediction(features)
        elif self.name == "texture_classifier":
            return self._texture_based_prediction(features)
        elif self.name == "color_classifier":
            return self._color_based_prediction(features)
        else:
            return self._default_prediction()
    
    def _abcde_based_prediction(self, features: Dict[str, float]) -> Dict[str, float]:
        """ABCDE-based heuristic classification."""
        a = features.get("asymmetry", 0.0)
        b = features.get("border_irregularity", 0.0) 
        c = features.get("color_variegation", 0.0)
        d = features.get("diameter_mm", 0.0)
        e = features.get("elevation_satellite", 0.0)
        
        # Heuristic scoring based on ABCDE criteria
        melanoma_score = 0.0
        if a > 0.35: melanoma_score += 0.2
        if b > 1.8: melanoma_score += 0.25
        if c >= 2.0: melanoma_score += 0.2
        if d > 6.0: melanoma_score += 0.15
        if e > 2.0: melanoma_score += 0.1
        
        # BCC tends to have regular borders but can be large
        bcc_score = 0.1
        if b < 1.3 and d > 4.0: bcc_score += 0.3
        if features.get("translucency", 0.0) > 0.5: bcc_score += 0.2
        
        # SCC has moderate irregularity
        scc_score = 0.05
        if 1.3 <= b <= 2.0 and d > 5.0: scc_score += 0.25
        if features.get("keratotic", 0.0) > 0.5: scc_score += 0.2
        
        # Nevus - generally regular
        nevus_score = 0.6 - melanoma_score
        if a < 0.2 and b < 1.3 and c < 1.5: nevus_score += 0.3
        
        # Seborrheic keratosis
        seb_k_score = 0.1
        if features.get("waxy_surface", 0.0) > 0.5: seb_k_score += 0.3
        
        # Normalize
        total = melanoma_score + bcc_score + scc_score + nevus_score + seb_k_score
        if total > 0:
            return {
                "melanoma": melanoma_score / total,
                "bcc": bcc_score / total,
                "scc": scc_score / total,
                "nevus": nevus_score / total,
                "seb_keratosis": seb_k_score / total
            }
        
        return self._default_prediction()
    
    def _texture_based_prediction(self, features: Dict[str, float]) -> Dict[str, float]:
        """Texture-based classification heuristics."""
        smoothness = features.get("surface_smoothness", 0.5)
        roughness = features.get("surface_roughness", 0.3)
        
        return {
            "melanoma": 0.15 + 0.1 * roughness,
            "bcc": 0.20 + 0.15 * smoothness,
            "scc": 0.18 + 0.12 * roughness,
            "nevus": 0.35 + 0.1 * smoothness,
            "seb_keratosis": 0.12 + 0.08 * roughness
        }
    
    def _color_based_prediction(self, features: Dict[str, float]) -> Dict[str, float]:
        """Color-based classification heuristics."""
        darkness = features.get("mean_darkness", 0.5)
        color_var = features.get("color_variegation", 0.0)
        
        melanoma_prob = 0.1 + 0.15 * color_var + 0.1 * darkness
        
        return {
            "melanoma": melanoma_prob,
            "bcc": 0.18 + 0.05 * (1 - darkness),
            "scc": 0.16 + 0.05 * darkness,
            "nevus": 0.4 - 0.1 * color_var,
            "seb_keratosis": 0.16 + 0.03 * color_var
        }
    
    def _default_prediction(self) -> Dict[str, float]:
        """Default conservative prediction."""
        return {
            "melanoma": 0.15,
            "bcc": 0.20,
            "scc": 0.15,
            "nevus": 0.40,
            "seb_keratosis": 0.10
        }

class EnsembleClassifier:
    """Ensemble classifier combining multiple models for robust diagnosis."""
    
    def __init__(self):
        self.classifiers = [
            LesionClassifier("abcde_classifier", weight=0.4),
            LesionClassifier("texture_classifier", weight=0.3),
            LesionClassifier("color_classifier", weight=0.3)
        ]
        self.diagnosis_classes = ["melanoma", "bcc", "scc", "nevus", "seb_keratosis"]
        
    def predict(self, features: Dict[str, float]) -> EnsembleOutput:
        """Generate ensemble prediction with uncertainty quantification."""
        
        # Collect predictions from all classifiers
        predictions = []
        weights = []
        
        for classifier in self.classifiers:
            try:
                pred = classifier.predict(features)
                predictions.append(pred)
                weights.append(classifier.weight)
            except Exception as e:
                logging.warning(f"Classifier {classifier.name} failed: {e}")
                continue
        
        if not predictions:
            # Fallback to conservative default
            return self._default_ensemble_output()
        
        # Compute weighted average
        avg_probs = self._compute_weighted_average(predictions, weights)
        
        # Calculate uncertainty metrics
        uncertainty_score = self._compute_uncertainty(predictions, avg_probs)
        
        # Generate confidence intervals using bootstrap-like approach
        confidence_intervals = self._compute_confidence_intervals(predictions, weights)
        
        # Create top-3 diagnoses
        top_diagnoses = self._get_top_diagnoses(avg_probs, confidence_intervals, uncertainty_score)
        
        # High-risk flagging
        high_risk_flag = self._assess_high_risk(avg_probs, uncertainty_score, features)
        
        # Feature explanations
        feature_explanations = self._generate_explanations(features, avg_probs)
        
        return EnsembleOutput(
            top_diagnoses=top_diagnoses,
            uncertainty_score=uncertainty_score,
            high_risk_flag=high_risk_flag,
            feature_explanations=feature_explanations
        )
    
    def _compute_weighted_average(self, predictions: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
        """Compute weighted average of predictions."""
        total_weight = sum(weights)
        avg_probs = {}
        
        for class_name in self.diagnosis_classes:
            weighted_sum = sum(pred.get(class_name, 0.0) * weight 
                             for pred, weight in zip(predictions, weights))
            avg_probs[class_name] = weighted_sum / total_weight
        
        return avg_probs
    
    def _compute_uncertainty(self, predictions: List[Dict[str, float]], avg_probs: Dict[str, float]) -> float:
        """Compute uncertainty score based on prediction variance."""
        if len(predictions) < 2:
            return 0.5  # High uncertainty for single model
        
        total_variance = 0.0
        for class_name in self.diagnosis_classes:
            class_preds = [pred.get(class_name, 0.0) for pred in predictions]
            variance = np.var(class_preds)
            total_variance += variance
        
        # Normalize uncertainty score to [0, 1]
        uncertainty = min(total_variance * 4, 1.0)  # Scale factor for interpretability
        return uncertainty
    
    def _compute_confidence_intervals(self, predictions: List[Dict[str, float]], weights: List[float]) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for each diagnosis."""
        confidence_intervals = {}
        
        for class_name in self.diagnosis_classes:
            class_preds = [pred.get(class_name, 0.0) for pred in predictions]
            if len(class_preds) >= 2:
                mean_pred = np.mean(class_preds)
                std_pred = np.std(class_preds)
                # 95% confidence interval approximation
                ci_lower = max(0.0, mean_pred - 1.96 * std_pred)
                ci_upper = min(1.0, mean_pred + 1.96 * std_pred)
                confidence_intervals[class_name] = (ci_lower, ci_upper)
            else:
                # Wide interval for single prediction
                pred_val = class_preds[0] if class_preds else 0.0
                confidence_intervals[class_name] = (max(0.0, pred_val - 0.2), min(1.0, pred_val + 0.2))
        
        return confidence_intervals
    
    def _get_top_diagnoses(self, avg_probs: Dict[str, float], 
                          confidence_intervals: Dict[str, Tuple[float, float]], 
                          uncertainty_score: float) -> List[DiagnosisResult]:
        """Get top-3 diagnoses with confidence metrics."""
        
        # Sort by probability
        sorted_diagnoses = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)
        
        top_diagnoses = []
        for i, (diagnosis, prob) in enumerate(sorted_diagnoses[:3]):
            ci = confidence_intervals.get(diagnosis, (prob - 0.1, prob + 0.1))
            
            # Confidence score based on probability magnitude and uncertainty
            confidence_score = prob * (1 - uncertainty_score)
            
            top_diagnoses.append(DiagnosisResult(
                diagnosis=diagnosis,
                probability=prob,
                confidence_interval=ci,
                confidence_score=confidence_score
            ))
        
        return top_diagnoses
    
    def _assess_high_risk(self, avg_probs: Dict[str, float], uncertainty_score: float, features: Dict[str, float]) -> bool:
        """Assess if case should be flagged as high-risk."""
        
        # High melanoma probability
        if avg_probs.get("melanoma", 0.0) >= 0.3:
            return True
        
        # High uncertainty with significant malignancy probability
        malignancy_prob = avg_probs.get("melanoma", 0.0) + avg_probs.get("scc", 0.0)
        if uncertainty_score > 0.6 and malignancy_prob > 0.2:
            return True
        
        # High-risk features
        if features.get("diameter_mm", 0.0) > 10.0:
            return True
        
        if features.get("border_irregularity", 0.0) > 2.5:
            return True
            
        return False
    
    def _generate_explanations(self, features: Dict[str, float], avg_probs: Dict[str, float]) -> Dict[str, float]:
        """Generate feature importance explanations."""
        explanations = {}
        
        # ABCDE feature contributions
        explanations["asymmetry_contribution"] = features.get("asymmetry", 0.0) * avg_probs.get("melanoma", 0.0)
        explanations["border_contribution"] = features.get("border_irregularity", 0.0) * 0.1
        explanations["color_contribution"] = features.get("color_variegation", 0.0) * 0.1
        explanations["diameter_contribution"] = min(features.get("diameter_mm", 0.0) / 10.0, 1.0)
        explanations["elevation_contribution"] = features.get("elevation_satellite", 0.0) * 0.05
        
        return explanations
    
    def _default_ensemble_output(self) -> EnsembleOutput:
        """Default conservative output when classifiers fail."""
        default_probs = {
            "nevus": 0.5,
            "melanoma": 0.2,
            "bcc": 0.15,
            "scc": 0.1,
            "seb_keratosis": 0.05
        }
        
        top_diagnoses = [
            DiagnosisResult("nevus", 0.5, (0.3, 0.7), 0.3),
            DiagnosisResult("melanoma", 0.2, (0.1, 0.4), 0.15),
            DiagnosisResult("bcc", 0.15, (0.05, 0.3), 0.1)
        ]
        
        return EnsembleOutput(
            top_diagnoses=top_diagnoses,
            uncertainty_score=0.7,  # High uncertainty
            high_risk_flag=True,    # Conservative flagging
            feature_explanations={}
        )