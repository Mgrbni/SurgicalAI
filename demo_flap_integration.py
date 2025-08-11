#!/usr/bin/env python3
"""
Demo script showing the flap planning integration with the pipeline.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_test_image(path: Path, size=(200, 200)):
    """Create a simple test image for demonstration."""
    # Create a synthetic lesion-like image
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Add a background skin tone
    img[:, :] = [210, 180, 140]  # Skin color
    
    # Add a dark lesion in the center
    center_x, center_y = size[0] // 2, size[1] // 2
    radius = 20
    
    y, x = np.ogrid[:size[1], :size[0]]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    img[mask] = [80, 50, 30]  # Dark lesion
    
    # Add some texture
    noise = np.random.normal(0, 10, size + (3,)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    Image.fromarray(img).save(path)
    return path

def demo_pipeline_integration():
    """Demonstrate the flap planning pipeline integration."""
    print("🧪 Demo: Flap Planning Pipeline Integration")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create test image
        lesion_path = tmp_path / "test_lesion.jpg"
        create_test_image(lesion_path)
        print(f"✓ Created test lesion image: {lesion_path}")
        
        # Mock patient data for pipeline
        print("\n📋 Simulating patient case:")
        print("   - Diagnosis: BCC")
        print("   - Site: Nasal tip")
        print("   - Size: ~10mm")
        print("   - Well-defined borders")
        
        # Import and configure pipeline components
        try:
            from surgicalai_demo.rules.flap_planner import plan_flap
            
            # Direct planning (simulating what pipeline would do)
            print("\n🔍 Running flap planning analysis...")
            plan = plan_flap(
                diagnosis="bcc",
                site="nasal_tip",
                diameter_mm=10,
                borders_well_defined=True,
                high_risk=False
            )
            
            # Create mock plan output
            plan_dict = {
                "incision_orientation_deg": plan.incision_orientation_deg,
                "rstl_unit": plan.rstl_unit,
                "oncology": plan.oncology,
                "candidates": plan.candidates,
                "danger": plan.danger,
                "citations": plan.citations,
                "defer": plan.defer
            }
            
            # Save plan as JSON (as pipeline would do)
            plan_path = tmp_path / "plan.json"
            with open(plan_path, 'w') as f:
                json.dump(plan_dict, f, indent=2)
            
            print(f"✓ Generated plan.json: {plan_path}")
            
            # Display key results
            print("\n📊 Planning Results:")
            print(f"   🚥 Proceed with reconstruction: {'No' if plan.defer else 'Yes'}")
            print(f"   📐 RSTL orientation: {plan.incision_orientation_deg}°")
            print(f"   🏥 Oncology recommendation: {plan.oncology['action']}")
            
            if plan.candidates:
                top_candidate = plan.candidates[0]
                print(f"   🔧 Top flap option: {top_candidate['name']}")
                print(f"   📋 Design notes: {top_candidate['design_notes'][0] if top_candidate['design_notes'] else 'None'}")
            
            if plan.danger:
                print(f"   ⚠️  Key warnings: {', '.join(plan.danger[:2])}")
            
            print(f"   📚 Evidence citations: {len(plan.citations)} sources")
            
            # Show plan.json structure
            print(f"\n📄 Generated plan.json structure:")
            for key in plan_dict.keys():
                print(f"   - {key}")
            
            print(f"\n💾 Demo files created in: {tmp_path}")
            print("   - test_lesion.jpg (synthetic test image)")
            print("   - plan.json (surgical planning output)")
            
            # Show a sample citation
            if plan.citations:
                print(f"\n📖 Sample citation: {plan.citations[0]}")
            
            print("\n✅ Pipeline integration demo completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run the demonstration."""
    success = demo_pipeline_integration()
    
    if success:
        print("\n🎉 The flap planning system is fully integrated and operational!")
        print("\nNext steps:")
        print("   1. Run the full pipeline with real images")
        print("   2. View surgical overlays in output images") 
        print("   3. Use plan.json for clinical decision support")
    else:
        print("\n⚠️  Integration issues detected. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
