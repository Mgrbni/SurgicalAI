from __future__ import annotations
from dotenv import load_dotenv; load_dotenv()
import os, json
from typing import Any, Dict
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".replace("OPENAI_", "OPENAI_"))  # tolerate typos
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", None)
TIMEOUT = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))

def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment (.env).")
    return OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT, timeout=TIMEOUT)

SYSTEM_MSG = (
    "You are a surgical decision-support assistant for dermatologic lesions. "
    "You NEVER give definitive diagnoses. You produce triage-oriented differentials with calibrated, "
    "round-number probabilities that sum to 100. You propose ONE primary local flap plan IF the site "
    "and size permit, referencing Langer’s lines and facial subunits. You add key caution notes. "
    "Output strictly as compact JSON matching the schema. No prose outside JSON."
)

SCHEMA_HINT = {
  "type":"object",
  "properties":{
    "triage_label":{"type":"string","enum":["BENIGN-LEANING","ATYPICAL","RED FLAG"]},
    "diagnosis_candidates":{"type":"array","items":{
      "type":"object","properties":{
        "name":{"type":"string"},
        "probability_pct":{"type":"number"}
      }, "required":["name","probability_pct"]
    }},
    "flap_plan":{
      "type":"object",
      "properties":{
        "indicated":{"type":"boolean"},
        "type":{"type":"string"},
        "design_summary":{"type":"string"},
        "tension_lines":{"type":"string"},
        "rotation_vector":{"type":"string"},
        "predicted_success_pct":{"type":"number"},
        "key_risks":{"type":"string"}
      },
      "required":["indicated","type","design_summary","tension_lines","rotation_vector","predicted_success_pct","key_risks"]
    },
    "notes":{"type":"string"}
  },
  "required":["triage_label","diagnosis_candidates","flap_plan","notes"]
}

def summarize_case(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    metrics = {label, risk_pct, rationale, asymmetry, border_irregularity, color_variegation,
               diameter_px, elevation_satellite, age, body_site}
    """
    prompt = {
        "role":"user",
        "content":(
            "Given the Tier-0 metrics below, produce JSON matching the schema. "
            "Calibrate probabilities with these hints: "
            "asymmetry↑, border_irregularity>1.3, color_variegation>=2, diameter_px>=60 favor melanoma risk; "
            "body_site and age adjust priors (face/cheek in older adults can raise suspicion). "
            "If lesion likely benign, flap_plan.indicated=false. "
            "If RED FLAG, suggest a conservative flap plan *only after biopsy margin clearance*, "
            "e.g., rotation flap, advancement, bilobed, rhomboid—choose ONE best fit for site. "
            f"\n\nSCHEMA_HINT={json.dumps(SCHEMA_HINT)}"
            f"\n\nMETRICS={json.dumps(metrics)}"
        )
    }

    client = _client()
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role":"system","content":SYSTEM_MSG}, prompt],
        max_output_tokens=700
    )
    text = resp.output_text  # SDK returns concatenated text
    try:
        data = json.loads(text)
    except Exception:
        # fallback: try to extract JSON between braces
        start = text.find("{"); end = text.rfind("}")
        data = json.loads(text[start:end+1]) if start!=-1 and end!=-1 else {"error":"LLM JSON parse failed","raw":text}
    return data
