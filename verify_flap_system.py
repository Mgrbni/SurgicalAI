#!/usr/bin/env python3
"""
Simple verification script for the flap planning system.
Tests key functionality without requiring complex setup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from surgicalai_demo.rules.flap_planner import plan_flap

def test_basic_planning():
    """Test basic flap planning functionality."""
    print("Testing basic flap planning...")
    
    # Test 1: BCC on nasal tip
    print("\n1. Testing BCC on nasal tip (10mm)...")
    plan = plan_flap("bcc", "nasal_tip", 10, borders_well_defined=True)
    
    print(f"   Defer: {plan.defer}")
    print(f"   RSTL orientation: {plan.incision_orientation_deg}°")
    print(f"   Oncology action: {plan.oncology['action']}")
    print(f"   Top candidate: {plan.candidates[0]['name'] if plan.candidates else 'None'}")
    print(f"   Dangers: {plan.danger[:2]}")  # First 2 dangers
    
    assert not plan.defer, "Low-risk BCC should not defer"
    assert plan.incision_orientation_deg == 45, "Nasal tip RSTL should be 45°"
    
    # Test 2: Melanoma (should defer)
    print("\n2. Testing melanoma (should defer)...")
    plan = plan_flap("melanoma", "cheek", 15)
    
    print(f"   Defer: {plan.defer}")
    print(f"   Reason: {plan.oncology['reason']}")
    
    assert plan.defer, "Melanoma should defer"
    assert 'staging' in plan.oncology['reason'].lower()
    
    # Test 3: Large SCC with ill-defined borders
    print("\n3. Testing high-risk SCC...")
    plan = plan_flap("scc", "cheek", 25, borders_well_defined=False)
    
    print(f"   Defer: {plan.defer}")
    print(f"   Action: {plan.oncology['action']}")
    
    assert plan.defer, "High-risk SCC should defer for Mohs"
    
    print("\n✅ All basic tests passed!")

def test_flap_selection():
    """Test flap selection logic."""
    print("\nTesting flap selection...")
    
    # Small nasal tip -> should prefer bilobed
    plan_small = plan_flap("bcc", "nasal_tip", 8, borders_well_defined=True)
    candidates_small = [c['flap'] for c in plan_small.candidates]
    
    # Large nasal tip -> should include forehead flap
    plan_large = plan_flap("bcc", "nasal_tip", 25, borders_well_defined=True)
    candidates_large = [c['flap'] for c in plan_large.candidates]
    
    print(f"   Small tip candidates: {candidates_small[:2]}")
    print(f"   Large tip candidates: {candidates_large[:2]}")
    
    assert any('bilobed' in c for c in candidates_small), "Should suggest bilobed for small tip"
    
    print("\n✅ Flap selection tests passed!")

def test_citations():
    """Test that citations are included."""
    print("\nTesting citations...")
    
    plan = plan_flap("bcc", "nasal_tip", 10)
    
    print(f"   Number of citations: {len(plan.citations)}")
    print(f"   Sample citations: {plan.citations[:2]}")
    
    assert len(plan.citations) > 0, "Should include citations"
    
    print("\n✅ Citation tests passed!")

def main():
    """Run all verification tests."""
    print("🏥 SurgicalAI Flap Planning System Verification")
    print("=" * 50)
    
    try:
        test_basic_planning()
        test_flap_selection() 
        test_citations()
        
        print("\n🎉 All verification tests passed!")
        print("The flap planning system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
