# Facial Flap Planning Rules

This document describes the evidence-based flap planning system integrated into SurgicalAI for facial reconstruction.

## Overview

The flap planning system provides source-backed recommendations for facial reconstruction based on:
- Oncologic safety gates
- RSTL (Relaxed Skin Tension Line) alignment
- Anatomic site considerations  
- Defect size and complexity
- Anatomic danger zones

## Oncologic Gates

All recommendations are filtered through oncologic safety gates to ensure appropriate sequencing of treatment:

### Melanoma
- **Action**: Always defer for staging
- **Margins**: 10-20mm based on Breslow depth (pending staging)
- **Recommendation**: Perform diagnostic biopsy → wide local excision → reconstruct after clear margins
- **Citations**: NCCN Melanoma Guidelines, MDPI Cancers 2021

### Basal Cell Carcinoma (BCC)

| Risk Level | Margins | Cure Rate | Recommendation |
|------------|---------|-----------|----------------|
| Low Risk | 4mm | 95% | Standard excision |
| High Risk | 5-10mm | 85% | Mohs surgery preferred |

**High-risk features**: H-zone location, recurrent, >2cm, ill-defined borders, aggressive histology

**H-zone**: nose, eyelids, ears, lips, chin, preauricular, postauricular regions

### Squamous Cell Carcinoma (SCC)

| Risk Level | Margins | Recommendation |
|------------|---------|----------------|
| Low Risk | 4-6mm | Standard excision |
| High Risk | 6-10mm | Mohs surgery preferred |

**High-risk features**: >2cm, >6mm depth, poorly differentiated, perineural invasion, immunosuppressed patient

## RSTL Alignment by Anatomic Unit

| Unit | RSTL Orientation | Notes |
|------|------------------|-------|
| Frontal | 90° (vertical) | Parallel to frontalis fibers |
| Glabella | 0° (horizontal) | Perpendicular to procerus/corrugator |
| Nasal dorsum | 90° (vertical) | Parallel to nasal framework |
| Nasal tip | 45° (oblique) | Complex 3D contours |
| Nasal ala | 30° (oblique) | Toward alar-facial groove |
| Upper eyelid | 0° (horizontal) | Hide in superior palpebral fold |
| Lower eyelid | 0° (horizontal) | Parallel to lid margin |
| Cheek (malar) | 30° (oblique) | Toward preauricular region |
| Cheek (buccal) | 15° (horizontal) | Toward ear |
| Upper lip | 90° (vertical) | Parallel to philtral columns |
| Lower lip | 90° (vertical) | Avoid horizontal scars |
| Chin | 0° (horizontal) | Parallel to mandible |
| Temple | 45° (oblique) | Toward hairline |

## Flap Selection Matrix

### Nasal Reconstruction

| Defect Size | Site | Primary Option | Secondary Options |
|-------------|------|----------------|-------------------|
| 5-15mm | Tip/Ala | **Bilobed flap** (Zitelli) | Nasolabial V-Y |
| 15-25mm | Tip/Dorsum | **Dorsal nasal flap** | Bilobed, Melolabial |
| >25mm | Any subunit | **Paramedian forehead flap** | Two-stage, consider subunit |

#### Bilobed Flap (Zitelli Design)
- **Indications**: Nasal tip/ala defects 8-15mm
- **Design**: First lobe 45-50°, second lobe 45-50°, total arc 90-100°
- **Proportions**: First lobe 50-60% defect diameter, second lobe 50% first lobe
- **Cautions**: Alar retraction, trapdoor deformity, tip bulkiness
- **Citations**: NCBI Facial Plastic Surgery 2020, Zitelli technique

#### Paramedian Forehead Flap
- **Indications**: Large nasal defects >15mm, multi-subunit involvement
- **Pedicle**: Supratrochlear artery (1.2cm from midline)
- **Staging**: Two-stage with pedicle division at 3 weeks
- **Planes**: Subgaleal, supraperiosteal
- **Cautions**: Forehead asymmetry, multiple procedures
- **Citations**: NCBI Plastic Surgery 2021, Iowa Head & Neck Protocols

### Eyelid Reconstruction

| Defect Size | Primary Option | Secondary Options |
|-------------|----------------|-------------------|
| <25% lid | **Primary closure** | Lateral cantholysis |
| 25-50% lid | **Tenzel semicircular** | Mustardé rotation |
| >50% lid | **Mustardé cheek rotation** | Hughes tarsoconjunctival |

#### Mustardé Cheek Rotation
- **Indications**: Lower eyelid defects 25-75% lid width
- **Design**: Large curvilinear incision into preauricular region
- **Vector**: Horizontal advancement to prevent ectropion
- **Support**: Frost suture for temporary lid support
- **Cautions**: Ectropion, facial nerve injury, lid malposition
- **Citations**: NCBI Ophthalmic Surgery 2021

### Cheek Reconstruction

| Defect Size | Primary Option | Design Notes |
|-------------|----------------|--------------|
| <15mm | **V-Y advancement** | Good for moderate laxity |
| 15-30mm | **Rotation flap** | Lateral cheek/temple |
| >30mm | **Cervicofacial advancement** | Sub-SMAS plane |

#### Cervicofacial Advancement
- **Indications**: Large cheek defects >25mm
- **Components**: Cervical and facial portions
- **Planes**: Sub-SMAS or subplatysmal  
- **Design**: Extensive undermining with SMAS imbrication
- **Cautions**: Facial nerve injury (all branches), distal necrosis
- **Citations**: Medicine LibreTexts 2021, PMC Annals Plastic Surgery

## Anatomic Danger Zones

### Facial Nerve Branches

| Branch | Location | Danger Zone | Safe Plane |
|--------|----------|-------------|------------|
| Temporal | Above zygomatic arch | Temple, lateral forehead | Subgaleal |
| Zygomatic | Crosses zygoma | Midface, cheek | Sub-SMAS |
| Buccal | Over masseter | Cheek, upper lip | Superficial to SMAS |
| Marginal mandibular | Mandibular border | Lower lip, chin | Subplatysmal |

### Vascular Structures
- **Supratrochlear artery**: 1.2cm from midline (forehead flap pedicle)
- **Angular artery**: Medial canthal region (nasal flap blood supply)
- **Facial artery**: Crosses mandible at masseter (cheek flap supply)

### Lacrimal System
- **Puncta/Canaliculi**: Medial canthal region
- **Risk**: Epiphora with >50% canalicular injury
- **Management**: Microsurgical repair with stenting

## Planning Algorithm

1. **Oncologic Gate Check**
   - High melanoma probability → defer for staging
   - High-risk BCC/SCC features → prefer Mohs
   - Ill-defined borders → margin-controlled excision

2. **RSTL Alignment**
   - Determine site-specific RSTL orientation
   - Plan incisions parallel to tension lines
   - Utilize scar-hiding anatomic folds

3. **Flap Selection**
   - Match defect size to flap capabilities
   - Consider anatomic site requirements
   - Evaluate tissue laxity and vascularity

4. **Danger Zone Assessment**
   - Identify nerve and vascular risks
   - Select appropriate surgical planes
   - Plan protective maneuvers

5. **Citation Integration**
   - Anchor recommendations to published evidence
   - Provide source references for patient discussion
   - Enable evidence-based consent process

## Implementation

The flap planning system is integrated into the SurgicalAI pipeline through:

- **Knowledge Base**: `surgicalai_demo/knowledge/facial_flaps.yaml`
- **Planning Engine**: `surgicalai_demo/rules/flap_planner.py`
- **Visual Overlays**: RSTL lines and flap designs on output images
- **Structured Output**: JSON plan with citations and recommendations

## Citations

1. **RSTL/Langer Lines**: ScienceDirect Archives Dermatological Research 2020
2. **Melanoma Margins**: NCBI Journal Surgical Oncology 2021, MDPI Cancers 2021
3. **BCC Margins**: PMC Dermatologic Surgery 2019, Medscape 2022
4. **SCC Guidelines**: NCCN Squamous Cell Skin Cancer 2023
5. **Bilobed Flap**: NCBI Facial Plastic Surgery 2020, Zitelli Design
6. **Paramedian Forehead**: NCBI Plastic Surgery 2021, Iowa Protocols
7. **Mustardé Flap**: NCBI Ophthalmic Surgery 2021
8. **Cervicofacial Flap**: Medicine LibreTexts 2021, PMC Annals 2020
9. **Facial Nerve Anatomy**: PubMed Aesthetic Surgery Journal 2021
10. **Nasolabial Flaps**: PMC Plastic Surgery International 2021

## Usage Example

```python
from surgicalai_demo.rules.flap_planner import plan_flap

# Plan reconstruction for BCC on nasal tip
plan = plan_flap(
    diagnosis="bcc",
    site="nasal_tip", 
    diameter_mm=10,
    borders_well_defined=True,
    high_risk=False
)

print(f"Defer reconstruction: {plan.defer}")
print(f"RSTL orientation: {plan.incision_orientation_deg}°")
print(f"Top candidate: {plan.candidates[0]['name']}")
print(f"Danger warnings: {plan.danger}")
```

This system provides evidence-based surgical planning recommendations while maintaining appropriate oncologic safety gates and anatomic considerations.
