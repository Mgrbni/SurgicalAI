"""
Prompt templates for surgical AI analysis.
"""

SYSTEM_PROMPT = """Return only valid JSON that exactly matches the provided schema fields and types. No prose, no Markdown, no backticks. If uncertain, set probabilities conservatively and include warnings."""

USER_PROMPT_TEMPLATE = """You are assisting a plastic surgery decision support demo.

Context:
- Lesion site: {{ site }}
- Extracted cues: {{ cues_json }}
- Contraindication flags: {{ flags_json }}

Task:
1) Produce ranking of likely diagnoses (e.g., melanoma, BCC, SCC, seborrheic keratosis, dermatofibroma, actinic keratosis).
2) Provide a primary_dx string.
3) If resection is indicated and safe, propose a single flap suggestion aligned to RSTL for the site.
4) Include 0–N contraindications and warnings.
5) Include citations array (short guideline refs).

Output:
Return **only** a JSON object compatible with AnalysisOutput schema:
{
  "diagnosis_probs": [{"label": "melanoma", "probability": 0.6}, ...],
  "primary_dx": "melanoma",
  "gradcam_notes": "High activation in irregular border regions",
  "reconstruction_plan": {
    "design": "Bilobed flap",
    "site": "nasal tip",
    "tension_axes": ["superior", "lateral"],
    "rationale": "Preserves alar symmetry",
    "predicted_success": 0.85
  },
  "contraindications": ["proximity to lacrimal system"],
  "warnings": ["consider Mohs for high-risk features"],
  "citations": ["NCCN Guidelines 2024", "Plastic Surgery Principles"]
}"""

VISION_PROMPT_TEMPLATE = """Analyze this dermatological lesion image for surgical planning.

Location: {{ site }}
Clinical context: {{ clinical_context }}

Provide structured analysis for surgical decision support:

1. Visual assessment of lesion characteristics
2. Differential diagnosis with probabilities
3. Reconstruction recommendations if applicable
4. Safety considerations and contraindications

Return only JSON matching the AnalysisOutput schema."""
