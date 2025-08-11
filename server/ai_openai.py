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

def summarize_case(full_metrics: dict) -> str:
    from openai import OpenAI
    import os
    client = OpenAI()

    SYSTEM_MSG = (
        "You are SurgicalAI, a cautious clinical summarizer. "
        "Write a brief, structured, non-diagnostic summary for a plastic surgeon. "
        "Flag uncertainty. Never give medical advice."
    )
    prompt = (
        "Summarize the following pipeline results for a clinical demo. "
        "Return 5 bullets: (1) Face/scan metadata, (2) Lesion detection probabilities, "
        "(3) Heatmap region(s), (4) Flap design candidates with rationale, "
        "(5) Risks/contraindications with confidence.\n\n"
        f"{full_metrics}"
    )

    try:
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=700,
        )
        return resp.output_text.strip()
    except Exception as e:
        # Dump the server’s JSON error if available — saves you 20 minutes
        try:
            from openai._exceptions import OpenAIError
            if isinstance(e, OpenAIError) and e.response is not None:
                raise RuntimeError(f"OpenAI error: {e.response.status_code} {e.response.text}") from e
        finally:
            raise
