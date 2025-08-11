"""
Vision-Language Model (VLM) Observer for lesion analysis.

This module provides AI-powered visual analysis of skin lesions using large
vision-language models (OpenAI GPT-4V, Anthropic Claude Vision).

Key Features:
- Provider-agnostic interface (OpenAI, Anthropic)
- Structured JSON output for integration with CNN predictions
- Dermatological terminology and clinical descriptors
- ABCD criteria estimation from visual inspection
- Recommendation generation based on visual findings

Configuration via environment variables:
- VLM_PROVIDER: "openai" or "anthropic" 
- OPENAI_API_KEY: Required for OpenAI provider
- ANTHROPIC_API_KEY: Required for Anthropic provider

NOT FOR CLINICAL USE - Research and demonstration purposes only.
"""
from __future__ import annotations

import os
import json
import time
import base64
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
from pathlib import Path
import warnings
import logging

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("OpenAI package not available. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    warnings.warn("Anthropic package not available. Install with: pip install anthropic")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Clinical prompt for structured lesion analysis
LESION_ANALYSIS_PROMPT = """You are an expert dermatologist analyzing a skin lesion image. Provide a structured analysis in JSON format only. Do not include any explanatory text outside the JSON.

Analyze the lesion and provide the following information:

1. Primary pattern recognition based on visual appearance
2. Likelihood estimates for each lesion type
3. Key descriptive features visible in the image
4. ABCD criteria assessment
5. Clinical recommendation based on visual findings

Respond with this exact JSON structure:

{
  "primary_pattern": "seborrheic_keratosis | melanoma | bcc | scc | nevus | other",
  "likelihoods": {
    "seborrheic_keratosis": 0.0-1.0,
    "melanoma": 0.0-1.0,
    "bcc": 0.0-1.0,
    "scc": 0.0-1.0,
    "nevus": 0.0-1.0
  },
  "descriptors": ["list of visual features observed"],
  "abcd_estimate": {
    "asymmetry": "low|moderate|high",
    "border": "sharp|irregular", 
    "color": "uniform|variegated",
    "diameter_mm": estimated_number,
    "evolution_suspected": true|false
  },
  "recommendation": "observe | dermoscopy | biopsy | mohs | wle"
}

Key descriptor vocabulary to use when applicable:
- "stuck-on appearance"
- "waxy surface"
- "sharply demarcated"
- "milia-like cysts"
- "comedo-like openings"
- "irregular pigment network"
- "blue-gray ovoid nests"
- "pearly rolled border"
- "ulceration"
- "asymmetric pigmentation"
- "variegated coloring"
- "atypical vascular pattern"

Focus on visible morphological features. If uncertain about any field, provide your best estimate rather than omitting it."""


class VLMObserver:
    """Vision-Language Model observer for dermatological analysis."""
    
    def __init__(self, provider: str = "openai", max_retries: int = 3, timeout: float = 30.0):
        """
        Initialize VLM observer.
        
        Args:
            provider: "openai" or "anthropic"
            max_retries: Maximum number of API retry attempts
            timeout: Request timeout in seconds
        """
        self.provider = provider.lower()
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize client based on provider
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package required. Install with: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            self.client = openai.OpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package required. Install with: pip install anthropic")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable required")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'")
    
    def describe_lesion_roi(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze lesion image and return structured JSON description.
        
        Args:
            image_bytes: Image data as bytes (JPEG or PNG)
            
        Returns:
            Dictionary with structured analysis or error information
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    return self._analyze_openai(image_bytes)
                elif self.provider == "anthropic":
                    return self._analyze_anthropic(image_bytes)
                    
            except Exception as e:
                logger.warning(f"VLM analysis attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return self._create_fallback_response(str(e))
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return self._create_fallback_response("Max retries exceeded")
    
    def _analyze_openai(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze image using OpenAI GPT-4V."""
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Detect image format
        if image_bytes.startswith(b'\xff\xd8'):
            media_type = "image/jpeg"
        elif image_bytes.startswith(b'\x89PNG'):
            media_type = "image/png"
        else:
            media_type = "image/jpeg"  # Default assumption
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": LESION_ANALYSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=800,
            temperature=0.1,  # Low temperature for consistent structured output
            timeout=self.timeout
        )
        
        content = response.choices[0].message.content.strip()
        return self._parse_json_response(content)
    
    def _analyze_anthropic(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze image using Anthropic Claude Vision."""
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Detect image format
        if image_bytes.startswith(b'\xff\xd8'):
            media_type = "image/jpeg"
        elif image_bytes.startswith(b'\x89PNG'):
            media_type = "image/png"
        else:
            media_type = "image/jpeg"
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": LESION_ANALYSIS_PROMPT
                        }
                    ]
                }
            ],
            timeout=self.timeout
        )
        
        content = response.content[0].text.strip()
        return self._parse_json_response(content)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate JSON response from VLM."""
        try:
            # Extract JSON from response (handle cases where model adds extra text)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = content[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['primary_pattern', 'likelihoods', 'descriptors', 'abcd_estimate', 'recommendation']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
            
            # Validate and normalize likelihoods
            if 'likelihoods' in data:
                likelihoods = data['likelihoods']
                total = sum(likelihoods.values())
                if total > 0:
                    # Normalize to sum to 1.0
                    data['likelihoods'] = {k: v/total for k, v in likelihoods.items()}
                    
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse VLM response: {e}")
            logger.error(f"Raw content: {content[:500]}...")
            return self._create_fallback_response(f"JSON parsing error: {e}")
    
    def _create_fallback_response(self, error_msg: str) -> Dict[str, Any]:
        """Create fallback response when VLM analysis fails."""
        return {
            "primary_pattern": "other",
            "likelihoods": {
                "seborrheic_keratosis": 0.2,
                "melanoma": 0.2,
                "bcc": 0.2,
                "scc": 0.2,
                "nevus": 0.2
            },
            "descriptors": ["analysis_unavailable"],
            "abcd_estimate": {
                "asymmetry": "moderate",
                "border": "irregular",
                "color": "variegated",
                "diameter_mm": 5.0,
                "evolution_suspected": False
            },
            "recommendation": "dermoscopy",
            "error": error_msg,
            "fallback": True
        }


def image_to_bytes(image: np.ndarray, format: str = "JPEG") -> bytes:
    """
    Convert numpy image array to bytes.
    
    Args:
        image: RGB image array [H, W, 3]
        format: Output format ("JPEG" or "PNG")
        
    Returns:
        Image as bytes
    """
    # Convert from RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # Encode to bytes
    if format.upper() == "JPEG":
        success, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif format.upper() == "PNG":
        success, buffer = cv2.imencode('.png', image_bgr)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    if not success:
        raise RuntimeError("Failed to encode image")
    
    return buffer.tobytes()


def describe_lesion_roi(
    image_bytes: bytes,
    provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for lesion analysis.
    
    Args:
        image_bytes: Image data as bytes
        provider: VLM provider ("openai" or "anthropic"). If None, uses VLM_PROVIDER env var.
        
    Returns:
        Structured analysis dictionary
    """
    if provider is None:
        provider = os.getenv("VLM_PROVIDER", "openai")
    
    try:
        observer = VLMObserver(provider=provider)
        return observer.describe_lesion_roi(image_bytes)
    except Exception as e:
        logger.error(f"VLM observation failed: {e}")
        return VLMObserver(provider="openai")._create_fallback_response(str(e))


def test_vlm_observer(test_image_path: Optional[str] = None):
    """Test VLM observer with a sample image."""
    if test_image_path is None:
        # Create a synthetic test image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        # Add a dark circular "lesion"
        cv2.circle(test_img, (112, 112), 50, (80, 60, 40), -1)
        test_bytes = image_to_bytes(test_img)
    else:
        with open(test_image_path, 'rb') as f:
            test_bytes = f.read()
    
    print("Testing VLM Observer...")
    
    providers = []
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    
    if not providers:
        print("No VLM providers available (missing API keys or packages)")
        return
    
    for provider in providers:
        print(f"\nTesting {provider.upper()} provider...")
        try:
            result = describe_lesion_roi(test_bytes, provider=provider)
            print(f"Analysis result:")
            print(json.dumps(result, indent=2))
            
            if result.get("fallback"):
                print(f"⚠️  Fallback response used due to: {result.get('error')}")
            else:
                print("✅ Successfully obtained VLM analysis")
                
        except Exception as e:
            print(f"❌ {provider} test failed: {e}")


if __name__ == "__main__":
    test_vlm_observer()
