# tests/test_flap_planner.py
"""
Unit tests for the facial flap planning system.
Tests oncologic gates, flap selection, and RSTL alignment.
"""

import pytest
from pathlib import Path
from surgicalai_demo.rules.flap_planner import plan_flap, FlapPlanner, FlapPlannerError


class TestFlapPlanner:
    """Test suite for the FlapPlanner class."""
    
    def test_bcc_nasal_tip_small(self):
        """Test BCC on nasal tip, small defect -> bilobed flap."""
        plan = plan_flap(
            diagnosis="bcc",
            site="nasal_tip", 
            diameter_mm=10,
            borders_well_defined=True,
            high_risk=False
        )
        
        assert not plan.defer, "Low-risk BCC should not defer"
        assert plan.oncology['action'] == 'Standard excision'
        assert plan.oncology['margins_mm'] == 4
        assert plan.incision_orientation_deg == 45  # nasal tip RSTL
        
        # Should prefer bilobed for small nasal tip
        candidate_names = [c['flap'] for c in plan.candidates]
        assert 'bilobed_zitelli' in candidate_names
        
        # Check for appropriate warnings
        assert any('alar retraction' in danger.lower() for danger in plan.danger)
    
    def test_melanoma_defer(self):
        """Test melanoma -> defer flap planning."""
        plan = plan_flap(
            diagnosis="melanoma",
            site="cheek",
            diameter_mm=18,
            borders_well_defined=True,
            high_risk=False
        )
        
        assert plan.defer, "Melanoma should defer reconstruction"
        assert 'WLE first' in plan.oncology['action']
        assert 'staging' in plan.oncology['reason'].lower()
        
        # Should still provide flap candidates for educational value
        assert len(plan.candidates) > 0
    
    def test_scc_large_lower_eyelid(self):
        """Test large SCC on lower eyelid -> cervicofacial with ectropion warning."""
        plan = plan_flap(
            diagnosis="scc",
            site="lower_eyelid",
            diameter_mm=25,
            borders_well_defined=False,  # High risk feature
            high_risk=False
        )
        
        assert plan.defer, "Ill-defined SCC should defer for Mohs"
        assert 'Mohs preferred' in plan.oncology['action']
        
        # Check RSTL alignment
        assert plan.incision_orientation_deg == 0  # horizontal for eyelid
        
        # Should suggest cervicofacial for large defects
        candidate_names = [c['flap'] for c in plan.candidates]
        assert any('cervicofacial' in name.lower() for name in candidate_names)
        
        # Check ectropion warning
        assert any('ectropion' in danger.lower() for danger in plan.danger)
    
    def test_bcc_high_risk_h_zone(self):
        """Test high-risk BCC in H-zone -> Mohs preferred."""
        plan = plan_flap(
            diagnosis="bcc",
            site="nasal_ala",  # H-zone
            diameter_mm=12,
            borders_well_defined=True,
            high_risk=True
        )
        
        assert plan.defer, "High-risk BCC should defer for Mohs"
        assert 'Mohs preferred' in plan.oncology['action']
        assert plan.oncology['margins_mm'] >= 5
    
    def test_site_normalization(self):
        """Test that site aliases are properly normalized."""
        planner = FlapPlanner()
        
        assert planner._normalize_site("tip") == "nasal_tip"
        assert planner._normalize_site("nose") == "nasal_tip" 
        assert planner._normalize_site("lid") == "lower_eyelid"
        assert planner._normalize_site("cheek") == "cheek_malar"
    
    def test_diagnosis_normalization(self):
        """Test that diagnosis aliases are properly normalized."""
        planner = FlapPlanner()
        
        assert planner._normalize_diagnosis("BCC") == "bcc"
        assert planner._normalize_diagnosis("Basal Cell Carcinoma") == "bcc"
        assert planner._normalize_diagnosis("SCC") == "scc"
        assert planner._normalize_diagnosis("Squamous Cell Carcinoma") == "scc"
        assert planner._normalize_diagnosis("Melanoma") == "melanoma"
    
    def test_flap_size_compatibility(self):
        """Test that flaps are only suggested for appropriate sizes."""
        # Small defect
        plan_small = plan_flap("bcc", "nasal_tip", 8, borders_well_defined=True)
        small_candidates = [c['flap'] for c in plan_small.candidates]
        
        # Large defect  
        plan_large = plan_flap("bcc", "nasal_tip", 25, borders_well_defined=True)
        large_candidates = [c['flap'] for c in plan_large.candidates]
        
        # Bilobed should be preferred for small, forehead for large
        assert 'bilobed_zitelli' in small_candidates
        assert any('forehead' in name for name in large_candidates)
    
    def test_citations_included(self):
        """Test that source citations are included in plans."""
        plan = plan_flap("bcc", "nasal_tip", 10, borders_well_defined=True)
        
        assert len(plan.citations) > 0
        # Should include some recognizable source patterns
        citation_text = ' '.join(plan.citations).lower()
        assert any(term in citation_text for term in ['ncbi', 'zitelli', 'bilobed', 'nasal'])
    
    def test_rstl_orientation(self):
        """Test RSTL orientation for different sites."""
        # Nasal tip should be oblique
        plan_tip = plan_flap("bcc", "nasal_tip", 10)
        assert plan_tip.incision_orientation_deg == 45
        
        # Eyelid should be horizontal  
        plan_lid = plan_flap("bcc", "lower_eyelid", 8)
        assert plan_lid.incision_orientation_deg == 0
        
        # Forehead should be vertical
        plan_forehead = plan_flap("bcc", "forehead", 15)
        assert plan_forehead.incision_orientation_deg == 90
    
    def test_invalid_knowledge_base(self):
        """Test handling of missing or invalid knowledge base."""
        with pytest.raises(FlapPlannerError):
            FlapPlanner(Path("nonexistent_file.yaml"))
    
    def test_unknown_diagnosis(self):
        """Test handling of unknown diagnosis."""
        plan = plan_flap("unknown_lesion", "cheek", 15)
        
        assert plan.defer, "Unknown diagnosis should defer"
        assert 'biopsy first' in plan.oncology['reason'].lower()
    
    def test_missing_diameter(self):
        """Test handling when diameter is not provided."""
        plan = plan_flap("bcc", "cheek", None, borders_well_defined=True)
        
        # Should still provide recommendations with default assumptions
        assert len(plan.candidates) > 0
        assert plan.incision_orientation_deg is not None


class TestFlapSelection:
    """Test specific flap selection logic."""
    
    def test_cheek_flap_progression(self):
        """Test flap selection for progressively larger cheek defects."""
        # Small cheek defect -> V-Y advancement
        plan_small = plan_flap("bcc", "cheek", 10, borders_well_defined=True)
        
        # Medium cheek defect -> rotation flap
        plan_medium = plan_flap("bcc", "cheek", 20, borders_well_defined=True)
        
        # Large cheek defect -> cervicofacial
        plan_large = plan_flap("bcc", "cheek", 35, borders_well_defined=True)
        
        # Check that different flaps are preferred for different sizes
        small_top = plan_small.candidates[0]['flap'] if plan_small.candidates else None
        medium_top = plan_medium.candidates[0]['flap'] if plan_medium.candidates else None  
        large_top = plan_large.candidates[0]['flap'] if plan_large.candidates else None
        
        # Should see progression in complexity/reach
        assert small_top != large_top, "Different flaps should be preferred for different sizes"
    
    def test_nasal_reconstruction_ladder(self):
        """Test reconstruction ladder for nasal defects."""
        # Small tip -> bilobed
        plan_small = plan_flap("bcc", "nasal_tip", 8, borders_well_defined=True)
        
        # Large tip -> forehead flap
        plan_large = plan_flap("bcc", "nasal_tip", 22, borders_well_defined=True)
        
        small_flaps = [c['flap'] for c in plan_small.candidates]
        large_flaps = [c['flap'] for c in plan_large.candidates]
        
        assert 'bilobed_zitelli' in small_flaps
        assert any('forehead' in flap for flap in large_flaps)


class TestOncologicGates:
    """Test oncologic safety gates."""
    
    def test_melanoma_always_defers(self):
        """Test that melanoma always defers regardless of size."""
        for size in [5, 15, 30]:
            plan = plan_flap("melanoma", "cheek", size)
            assert plan.defer, f"Melanoma {size}mm should defer"
    
    def test_bcc_risk_stratification(self):
        """Test BCC risk stratification affects recommendations."""
        # Low risk BCC
        plan_low = plan_flap("bcc", "cheek", 15, borders_well_defined=True, high_risk=False)
        
        # High risk BCC (ill-defined borders)
        plan_high = plan_flap("bcc", "cheek", 15, borders_well_defined=False, high_risk=False)
        
        assert not plan_low.defer, "Low-risk BCC should not defer"
        assert plan_high.defer, "High-risk BCC should defer for Mohs"
    
    def test_size_based_risk(self):
        """Test that large lesions trigger high-risk pathway."""
        # Large BCC should defer even if otherwise low-risk
        plan = plan_flap("bcc", "cheek", 30, borders_well_defined=True, high_risk=False)
        
        # Size >20mm should trigger high-risk pathway
        assert plan.defer, "Large BCC should defer for margin control"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
