# surgicalai_demo/rules/flap_planner.py
"""
Source-backed flap planning engine for facial reconstruction.
Integrates oncologic margins, RSTL alignment, and evidence-based flap selection.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class Plan:
    """Structured flap planning result."""
    incision_orientation_deg: Optional[float]
    rstl_unit: str
    oncology: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    danger: List[str]
    citations: List[str]
    defer: bool = False


class FlapPlannerError(Exception):
    """Custom exception for flap planning errors."""
    pass


class FlapPlanner:
    """Evidence-based facial flap planning system."""
    
    def __init__(self, knowledge_path: Optional[Path] = None):
        """Initialize with knowledge base."""
        if knowledge_path is None:
            # Default to knowledge directory relative to this file
            knowledge_path = Path(__file__).parent.parent / "knowledge" / "facial_flaps.yaml"
        
        self.knowledge_path = Path(knowledge_path)
        self._load_knowledge()
    
    def _load_knowledge(self) -> None:
        """Load the facial flaps knowledge base."""
        if not self.knowledge_path.exists():
            raise FlapPlannerError(f"Knowledge base not found: {self.knowledge_path}")
        
        try:
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                self.knowledge = yaml.safe_load(f)
        except Exception as e:
            raise FlapPlannerError(f"Failed to load knowledge base: {e}")
        
        # Validate required sections
        required_sections = ['rstl_lines', 'flap_options', 'oncologic_margins', 'danger_zones', 'sources']
        for section in required_sections:
            if section not in self.knowledge:
                raise FlapPlannerError(f"Missing required section in knowledge base: {section}")
    
    def _normalize_site(self, site: str) -> str:
        """Normalize anatomic site names."""
        site_map = {
            'tip': 'nasal_tip',
            'nose': 'nasal_tip',
            'nostril': 'nasal_ala', 
            'ala': 'nasal_ala',
            'dorsum': 'nasal_dorsum',
            'bridge': 'nasal_dorsum',
            'lid': 'lower_eyelid',
            'eyelid': 'lower_eyelid',
            'cheek': 'cheek_malar',
            'chin': 'chin',
            'lip': 'upper_lip',
            'forehead': 'frontal',
            'temple': 'temple'
        }
        
        normalized = site.lower().strip().replace(' ', '_')
        return site_map.get(normalized, normalized)
    
    def _normalize_diagnosis(self, diagnosis: str) -> str:
        """Normalize diagnosis names."""
        dx_map = {
            'melanoma': 'melanoma',
            'bcc': 'bcc', 
            'basal cell carcinoma': 'bcc',
            'scc': 'scc',
            'squamous cell carcinoma': 'scc',
            'squamous cell': 'scc'
        }
        
        normalized = diagnosis.lower().strip()
        return dx_map.get(normalized, normalized)
    
    def _check_oncologic_gate(
        self, 
        diagnosis: str, 
        diameter_mm: Optional[float],
        high_risk: bool = False,
        borders_well_defined: bool = True
    ) -> Dict[str, Any]:
        """Check oncologic appropriateness for immediate reconstruction."""
        
        dx = self._normalize_diagnosis(diagnosis)
        margins_data = self.knowledge['oncologic_margins']
        
        # Default to defer for safety
        result = {
            'action': 'defer',
            'margins_mm': 10,
            'reason': 'Unknown diagnosis - biopsy first',
            'defer': True
        }
        
        if dx == 'melanoma':
            # Always defer melanoma unless explicitly staging known
            result.update({
                'action': 'WLE first',
                'margins_mm': 10,  # Conservative pending Breslow
                'reason': 'Melanoma - require staging before reconstruction',
                'defer': True
            })
            
        elif dx == 'bcc':
            bcc_data = margins_data['bcc']
            
            # Check if high-risk features present
            is_high_risk = (
                high_risk or 
                not borders_well_defined or
                (diameter_mm and diameter_mm > 20)
            )
            
            if is_high_risk:
                result.update({
                    'action': 'Mohs preferred',
                    'margins_mm': bcc_data['high_risk']['margins_mm'][0],
                    'reason': 'High-risk BCC features',
                    'defer': True
                })
            else:
                result.update({
                    'action': 'Standard excision',
                    'margins_mm': bcc_data['low_risk']['margins_mm'][0],
                    'reason': 'Low-risk BCC - standard margins',
                    'defer': False
                })
                
        elif dx == 'scc':
            scc_data = margins_data['scc']
            
            is_high_risk = (
                high_risk or
                not borders_well_defined or
                (diameter_mm and diameter_mm > 20)
            )
            
            if is_high_risk:
                result.update({
                    'action': 'Mohs preferred',
                    'margins_mm': scc_data['high_risk']['margins_mm'][0],
                    'reason': 'High-risk SCC features',
                    'defer': True
                })
            else:
                result.update({
                    'action': 'Standard excision',
                    'margins_mm': scc_data['low_risk']['margins_mm'][0],
                    'reason': 'Low-risk SCC - standard margins',
                    'defer': False
                })
        
        return result
    
    def _get_rstl_info(self, site: str) -> Dict[str, Any]:
        """Get RSTL orientation and scar hiding information for site."""
        site = self._normalize_site(site)
        rstl_data = self.knowledge['rstl_lines']
        
        # Find best match for site
        rstl_info = None
        for rstl_site, data in rstl_data.items():
            if site in rstl_site or rstl_site in site:
                rstl_info = data
                break
        
        # Default if no match found
        if rstl_info is None:
            rstl_info = {
                'rstl_axis_deg': 0,  # Default horizontal
                'notes': 'Site-specific RSTL not defined',
                'scar_hiding_folds': ['natural skin creases'],
                'source_id': 'default'
            }
        
        return rstl_info
    
    def _select_flap_candidates(
        self, 
        site: str, 
        diameter_mm: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Select appropriate flap candidates based on site and size."""
        
        site = self._normalize_site(site)
        flap_data = self.knowledge['flap_options']
        candidates = []
        
        # Use conservative diameter if not specified
        if diameter_mm is None:
            diameter_mm = 15.0  # Default moderate size
        
        for flap_name, flap_info in flap_data.items():
            fit_score = self._evaluate_flap_fit(flap_info, site, diameter_mm)
            
            if fit_score > 0:
                candidate = {
                    'flap': flap_name,
                    'name': flap_info['name'],
                    'fit': self._categorize_fit(fit_score),
                    'fit_score': fit_score,
                    'design_notes': flap_info.get('design_notes', []),
                    'danger': flap_info.get('danger', []),
                    'vascular': flap_info.get('vascular', {}),
                    'planes': flap_info.get('planes', [])
                }
                
                # Add specific design parameters if available
                if 'angles' in flap_info:
                    candidate['vector'] = flap_info['angles']
                
                candidates.append(candidate)
        
        # Sort by fit score (higher is better)
        candidates.sort(key=lambda x: x['fit_score'], reverse=True)
        
        return candidates[:5]  # Return top 5 candidates
    
    def _evaluate_flap_fit(self, flap_info: Dict, site: str, diameter_mm: float) -> float:
        """Evaluate how well a flap fits the defect requirements."""
        
        score = 0.0
        
        # Check size compatibility
        size_limits = flap_info.get('size_limits_mm', {})
        min_size = size_limits.get('min', 0)
        max_size = size_limits.get('max', 100)
        
        if min_size <= diameter_mm <= max_size:
            score += 3.0  # Base score for size compatibility
            
            # Bonus for optimal size range
            optimal_range = (max_size - min_size) * 0.3
            if min_size + optimal_range <= diameter_mm <= max_size - optimal_range:
                score += 1.0
        else:
            return 0.0  # Not suitable if size incompatible
        
        # Check anatomic site compatibility
        indications = flap_info.get('indications', [])
        for indication in indications:
            if site in indication.lower() or any(keyword in indication.lower() for keyword in site.split('_')):
                score += 2.0
                break
        
        # Site-specific scoring
        if site.startswith('nasal'):
            if 'nasal' in flap_info['name'].lower():
                score += 1.5
            if 'forehead' in flap_info['name'].lower() and diameter_mm > 15:
                score += 1.0
            if 'bilobed' in flap_info['name'].lower() and diameter_mm <= 15:
                score += 1.5
                
        elif site.startswith('cheek'):
            if 'cheek' in flap_info['name'].lower():
                score += 1.5
            if 'cervicofacial' in flap_info['name'].lower() and diameter_mm > 25:
                score += 1.0
                
        elif 'eyelid' in site:
            if 'mustarde' in flap_info['name'].lower():
                score += 1.5
            if 'advancement' in flap_info['name'].lower() and diameter_mm <= 15:
                score += 1.0
        
        return score
    
    def _categorize_fit(self, score: float) -> str:
        """Categorize the fit quality."""
        if score >= 5.0:
            return 'excellent'
        elif score >= 4.0:
            return 'good'
        elif score >= 3.0:
            return 'fair'
        elif score >= 2.0:
            return 'poor'
        else:
            return 'unsuitable'
    
    def _collect_anatomic_dangers(self, site: str, flap_candidates: List[Dict]) -> List[str]:
        """Collect relevant anatomic danger warnings."""
        
        dangers = set()
        site = self._normalize_site(site)
        danger_data = self.knowledge['danger_zones']
        
        # Site-specific dangers
        if 'nasal' in site:
            dangers.add('angular artery at medial canthus')
            dangers.add('alar retraction risk')
            
        if 'eyelid' in site:
            dangers.add('ectropion from vertical tension')
            dangers.add('lacrimal system injury')
            
        if 'cheek' in site:
            dangers.add('facial nerve branches')
            dangers.add('facial artery injury')
            
        if 'temple' in site:
            dangers.add('temporal branch facial nerve')
            dangers.add('superficial temporal artery')
        
        # Add flap-specific dangers
        for candidate in flap_candidates:
            dangers.update(candidate.get('danger', []))
        
        return sorted(list(dangers))
    
    def _collect_citations(self, flap_candidates: List[Dict], rstl_info: Dict) -> List[str]:
        """Collect source citations for the plan."""
        
        citations = set()
        
        # Add RSTL source
        rstl_source = rstl_info.get('source_id')
        if rstl_source:
            citations.add(rstl_source)
        
        # Add flap sources
        flap_data = self.knowledge['flap_options']
        for candidate in flap_candidates:
            flap_name = candidate['flap']
            if flap_name in flap_data:
                flap_sources = flap_data[flap_name].get('source_ids', [])
                citations.update(flap_sources)
        
        # Convert to source details
        source_details = []
        sources = self.knowledge.get('sources', {})
        for citation_id in citations:
            if citation_id in sources:
                source = sources[citation_id]
                detail = f"{source.get('title', citation_id)} ({source.get('year', 'Unknown')})"
                source_details.append(detail)
        
        return sorted(source_details)


def plan_flap(
    diagnosis: str,
    site: str, 
    diameter_mm: Optional[float],
    borders_well_defined: bool = True,
    high_risk: bool = False,
    knowledge_path: Optional[Path] = None
) -> Plan:
    """
    Main flap planning function.
    
    Args:
        diagnosis: Primary diagnosis (melanoma, bcc, scc, etc.)
        site: Anatomic location (nasal_tip, cheek, lower_eyelid, etc.)
        diameter_mm: Lesion diameter in mm
        borders_well_defined: Whether lesion borders are well-defined
        high_risk: Whether high-risk features present
        knowledge_path: Optional path to knowledge base
    
    Returns:
        Plan object with reconstruction recommendations
    """
    
    planner = FlapPlanner(knowledge_path)
    
    # 1. Oncologic gate check
    oncology = planner._check_oncologic_gate(
        diagnosis, diameter_mm, high_risk, borders_well_defined
    )
    
    # 2. Get RSTL information
    rstl_info = planner._get_rstl_info(site)
    
    # 3. Select flap candidates (even if deferred, for educational value)
    candidates = planner._select_flap_candidates(site, diameter_mm)
    
    # 4. Collect anatomic dangers
    dangers = planner._collect_anatomic_dangers(site, candidates)
    
    # 5. Collect citations
    citations = planner._collect_citations(candidates, rstl_info)
    
    # 6. Create plan
    plan = Plan(
        incision_orientation_deg=rstl_info['rstl_axis_deg'],
        rstl_unit=planner._normalize_site(site),
        oncology=oncology,
        candidates=candidates,
        danger=dangers,
        citations=citations,
        defer=oncology['defer']
    )
    
    return plan


# Convenience function for backward compatibility
def plan_reconstruction(*args, **kwargs) -> Dict[str, Any]:
    """Legacy wrapper for existing pipeline integration."""
    # Map old arguments to new function
    probs = kwargs.get('probs', {})
    location = kwargs.get('location', 'cheek')
    diameter = kwargs.get('defect_equiv_diam_mm')
    
    # Extract top diagnosis from probabilities
    if probs:
        top_dx = max(probs.items(), key=lambda x: x[1])[0] if probs else 'unknown'
    else:
        top_dx = 'unknown'
    
    try:
        plan = plan_flap(
            diagnosis=top_dx,
            site=location,
            diameter_mm=diameter,
            borders_well_defined=not kwargs.get('ill_defined_borders', False),
            high_risk=kwargs.get('recurrent_tumor', False)
        )
        
        # Convert Plan object to dict for compatibility
        return {
            'indicated': not plan.defer,
            'reason': plan.oncology['reason'],
            'guidance': f"RSTL orientation: {plan.incision_orientation_deg}°",
            'citations': plan.citations,
            'options': [
                {
                    'name': c['name'],
                    'rationale': f"Fit: {c['fit']} for {location}",
                    'success_hint': ', '.join(c['design_notes'][:2]),
                    'cautions': c['danger'][:3],
                    'priority': idx + 1
                }
                for idx, c in enumerate(plan.candidates[:3])
            ]
        }
        
    except Exception as e:
        # Fallback for errors
        return {
            'indicated': False,
            'reason': f'Planning error: {str(e)}',
            'guidance': 'Manual consultation recommended',
            'citations': [],
            'options': []
        }
