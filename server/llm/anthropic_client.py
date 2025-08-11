"""
Anthropic client implementation for surgical AI analysis.
"""
import base64
import logging
import time
from typing import Dict, Any, Iterable, Optional, Union

import anthropic
import orjson
from pydantic import BaseModel, ValidationError

from .base import LLMClient, LLMResponse, LLMUsage, LLMError, LLMValidationError, LLMTimeoutError, LLMRateLimitError

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic client with structured JSON output support"""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "claude-3-5-sonnet-20241022",
        timeout_seconds: int = 45
    ):
        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_seconds
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        
        # Rough cost estimates (USD per 1M tokens)
        self.cost_per_1m_input = {
            "claude-3-5-sonnet": 3.0,
            "claude-3-5-haiku": 0.25,
            "claude-3-opus": 15.0,
            "claude-3-sonnet": 3.0,
            "claude-3-haiku": 0.25
        }
        self.cost_per_1m_output = {
            "claude-3-5-sonnet": 15.0,
            "claude-3-5-haiku": 1.25,
            "claude-3-opus": 75.0,
            "claude-3-sonnet": 15.0,
            "claude-3-haiku": 1.25
        }
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def _estimate_cost(self, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
        """Estimate cost in USD"""
        if not input_tokens and not output_tokens:
            return None
            
        # Extract base model name
        model_key = "claude-3"
        if "claude-3-5" in self.model:
            if "sonnet" in self.model:
                model_key = "claude-3-5-sonnet"
            elif "haiku" in self.model:
                model_key = "claude-3-5-haiku"
        elif "opus" in self.model:
            model_key = "claude-3-opus"
        elif "sonnet" in self.model:
            model_key = "claude-3-sonnet"
        elif "haiku" in self.model:
            model_key = "claude-3-haiku"
        
        cost = 0.0
        if input_tokens and model_key in self.cost_per_1m_input:
            cost += (input_tokens / 1_000_000) * self.cost_per_1m_input[model_key]
        if output_tokens and model_key in self.cost_per_1m_output:
            cost += (output_tokens / 1_000_000) * self.cost_per_1m_output[model_key]
            
        return cost if cost > 0 else None
    
    def _prepare_messages(
        self, 
        user: str, 
        image_data: Optional[bytes] = None,
        image_mime_type: Optional[str] = None
    ) -> list[dict]:
        """Prepare messages for Anthropic API"""
        if image_data:
            # Encode image for vision
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            content = [
                {"type": "text", "text": user},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_mime_type or "image/jpeg",
                        "data": encoded_image
                    }
                }
            ]
        else:
            content = user
            
        return [{"role": "user", "content": content}]
    
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_output_tokens: int,
        stream: bool = False,
        image_data: Optional[bytes] = None,
        image_mime_type: Optional[str] = None
    ) -> Union[LLMResponse, Iterable[str]]:
        """Complete with JSON output validated against schema"""
        start_time = time.time()
        
        messages = self._prepare_messages(user, image_data, image_mime_type)
        
        # Add JSON format instruction to system prompt
        enhanced_system = f"{system}\n\nProvide response as valid JSON matching this schema: {schema.model_json_schema()}"
        
        try:
            if stream:
                return self._stream_completion(enhanced_system, messages, max_output_tokens, start_time, schema)
            else:
                return self._non_stream_completion(enhanced_system, messages, max_output_tokens, start_time, schema)
                
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit: {e}")
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(f"Anthropic timeout: {e}")
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}")
    
    def _non_stream_completion(
        self, 
        system: str,
        messages: list[dict], 
        max_output_tokens: int, 
        start_time: float,
        schema: type[BaseModel]
    ) -> LLMResponse:
        """Non-streaming completion"""
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=0.1
            )
            
            duration = time.time() - start_time
            content = response.content[0].text if response.content else ""
            
            if not content:
                raise LLMValidationError("Empty response from Anthropic")
            
            # Parse and validate JSON
            try:
                parsed_json = orjson.loads(content)
                validated_data = schema.model_validate(parsed_json)
                result_dict = validated_data.model_dump()
            except orjson.JSONDecodeError as e:
                raise LLMValidationError(f"Invalid JSON from Anthropic: {e}", content[:500])
            except ValidationError as e:
                raise LLMValidationError(f"Schema validation failed: {e}", content[:500])
            
            # Extract usage info
            input_tokens = response.usage.input_tokens if response.usage else None
            output_tokens = response.usage.output_tokens if response.usage else None
            cost = self._estimate_cost(input_tokens, output_tokens)
            
            usage = LLMUsage(
                provider=self.provider_name,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_seconds=duration,
                meta={"stop_reason": response.stop_reason}
            )
            
            return LLMResponse(
                content=result_dict,
                usage=usage,
                raw_response=content
            )
            
        except Exception as e:
            if isinstance(e, (LLMValidationError, LLMError)):
                raise
            raise LLMError(f"Anthropic completion failed: {e}")
    
    def _stream_completion(
        self, 
        system: str,
        messages: list[dict], 
        max_output_tokens: int, 
        start_time: float,
        schema: type[BaseModel]
    ) -> Iterable[str]:
        """Streaming completion - yields text chunks"""
        try:
            accumulated_content = ""
            
            with self.client.messages.stream(
                model=self.model,
                system=system,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=0.1
            ) as stream:
                for text in stream.text_stream:
                    accumulated_content += text
                    yield text
            
            # Final validation (though client will handle this)
            if accumulated_content:
                try:
                    parsed_json = orjson.loads(accumulated_content)
                    schema.model_validate(parsed_json)
                except (orjson.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Stream validation failed: {e}")
                    
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit: {e}")
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(f"Anthropic timeout: {e}")
        except anthropic.APIError as e:
            raise LLMError(f"Anthropic API error: {e}")
        except Exception as e:
            raise LLMError(f"Anthropic streaming failed: {e}")
