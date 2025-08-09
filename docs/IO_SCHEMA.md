# IO Schema

This document outlines JSON/NPZ structures used in the prototype.

## mesh.npz
`{"vertices": (N,3), "faces": (M,3)}`

## lesion_probs.json
`{"class_probs":{"melanoma":float,"benign":float},"top_class":str}`

## flap_plan.json
`{"type":"rotation","pivot":[x,y,z],"arc":[[x,y,z],...],"tension_axis":[ux,uy,uz],"success_prob":float,"notes":str}`

## contraindications.json
`{"alerts":[{"type":str,"distance_mm":float,"details":str}],"score":float}`

## narrative.json
`{"summary":str,"risk_explanation":str,"flap_rationale":str,"alternatives":[{"approach":str,"when":str}],"disclaimer":str}`
