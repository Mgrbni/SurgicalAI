"""LLM integration for SurgicalAI with provider routing (OpenAI/Anthropic) and offline fallback."""

import os
import logging
from typing import Optional, Dict, Any, List
from PIL import Image

from .schemas import AnalysisRequest, Diagnostics, DiagnosisProb
from .utils import image_to_base64


logger = logging.getLogger(__name__)


async def get_llm_analysis(roi_image: Image.Image, request: AnalysisRequest, request_id: str) -> Diagnostics:
    """Route to the selected LLM provider (OpenAI/Anthropic)."""
    if request.offline or get_offline_mode():
        logger.info("LLM analysis skipped (offline mode)")
        raise Exception("LLM unavailable")

    # Compose context
    context_parts: List[str] = []
    if getattr(request, 'site', None):
        context_parts.append(f"Anatomical location: {request.site}")
    rf = getattr(request, 'risk_factors', {}) or {}
    if isinstance(rf, dict):
        if rf.get('h_zone'):
            context_parts.append("Located in H-zone (high-risk area)")
        if rf.get('ill_defined_borders'):
            context_parts.append("Ill-defined borders noted")
        if rf.get('recurrent_tumor') or rf.get('recurrent'):
            context_parts.append("Recurrent tumor")
        if rf.get('age'):
            context_parts.append(f"Patient age: {rf['age']}")
    context = ". ".join(context_parts) if context_parts else "No additional context provided."

    # Image base64
    image_b64 = image_to_base64(roi_image, "JPEG")

    provider = (getattr(request, 'provider', None) or 'openai').lower()
    model = getattr(request, 'model', None) or ("gpt-4o" if provider == 'openai' else "claude-3-sonnet-20240229")

    if provider == 'anthropic':
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise Exception("Anthropic API key missing")
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=api_key)
            system_prompt = (
                "You are an expert dermatologist. Return STRICT JSON with keys top3 (array of 3 objects with label and prob) and notes. "
                "Only output JSON. Probabilities must sum to ~1.0."
            )
            user_text = f"Analyze this skin lesion. Clinical context: {context}"
            msg = await client.messages.create(
                model=model,
                max_tokens=800,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                        ],
                    }
                ],
            )
            # Concatenate text blocks
            content_text = "".join([b.text for b in msg.content if getattr(b, 'type', '') == 'text'])
            import json as _json
            data = _json.loads(content_text)
            top3 = [DiagnosisProb(label=item["label"], prob=float(item["prob"])) for item in data.get("top3", [])][:3]
            if len(top3) < 3:
                # pad with generic
                while len(top3) < 3:
                    top3.append(DiagnosisProb(label="nevus", prob=0.0))
            diag = Diagnostics(top3=top3, notes=data.get("notes", ""))
            logger.info(f"LLM (Anthropic) analysis completed for request {request_id}")
            return diag
        except Exception as e:
            logger.error(f"Anthropic analysis failed for request {request_id}: {e}")
            raise

    # Default: OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OpenAI API key missing")
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=api_key)

        diagnosis_schema = {
            "type": "object",
            "properties": {
                "top3": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "prob": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["label", "prob"],
                    },
                    "minItems": 3,
                    "maxItems": 3,
                },
                "notes": {"type": "string"},
            },
            "required": ["top3", "notes"],
        }

        response = await client.chat.completions.create(
            model=model,
            max_completion_tokens=800,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert dermatologist analyzing skin lesions. "
                        "Provide differential diagnosis with probabilities for the top 3 most likely conditions. "
                        "Use standard medical terminology. Probabilities should sum to 1.0. Provide brief clinical reasoning in notes."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this skin lesion. Clinical context: {context}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"},
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "lesion_diagnosis", "schema": diagnosis_schema},
            },
        )
        import json as _json
        data = _json.loads(response.choices[0].message.content)
        top3 = [DiagnosisProb(label=item["label"], prob=float(item["prob"])) for item in data["top3"]]
        diagnostics = Diagnostics(top3=top3, notes=data.get("notes", ""))
        logger.info(f"LLM (OpenAI) analysis completed for request {request_id}")
        return diagnostics
    except Exception as e:
        logger.error(f"OpenAI analysis failed for request {request_id}: {e}")
        raise


def get_offline_mode() -> bool:
    """Check if we should run in offline mode."""
    return os.getenv("OFFLINE_ANALYSIS", "false").lower() == "true"
