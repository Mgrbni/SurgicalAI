"""
OpenAI client implementation for surgical AI analysis.
"""
import asyncio
import base64
import logging
import time
from typing import Dict, Any, Iterable, Optional, Union

import openai
import orjson
from pydantic import BaseModel, ValidationError

from .base import LLMClient, LLMResponse, LLMUsage, LLMError, LLMValidationError, LLMTimeoutError, LLMRateLimitError

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI client with structured JSON output support"""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o-mini",
        timeout_seconds: int = 45,
        project: Optional[str] = None,
        organization: Optional[str] = None
    ):
        self.client = openai.OpenAI(
            api_key=api_key,
            project=project,
            organization=organization,
            timeout=timeout_seconds
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        
        # Rough cost estimates (USD per 1K tokens)
        self.cost_per_1k_input = {
            "gpt-4o": 0.0025,
            "gpt-4o-mini": 0.00015,
            "gpt-4-turbo": 0.01,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.0015
        }
        self.cost_per_1k_output = {
            "gpt-4o": 0.01,
            "gpt-4o-mini": 0.0006,
            "gpt-4-turbo": 0.03,
            "gpt-4": 0.06,
            "gpt-3.5-turbo": 0.002
        }
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def model_name(self) -> str:
        return self.model
    
    def _estimate_cost(self, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
        """Estimate cost in USD"""
        if not input_tokens and not output_tokens:
            return None
            
        model_base = self.model.split("-")[0:2]  # e.g., "gpt-4o" from "gpt-4o-mini"
        model_key = "-".join(model_base)
        
        cost = 0.0
        if input_tokens and model_key in self.cost_per_1k_input:
            cost += (input_tokens / 1000) * self.cost_per_1k_input[model_key]
        if output_tokens and model_key in self.cost_per_1k_output:
            cost += (output_tokens / 1000) * self.cost_per_1k_output[model_key]
            
        return cost if cost > 0 else None
    
    def _prepare_messages(
        self, 
        system: str, 
        user: str, 
        image_data: Optional[bytes] = None,
        image_mime_type: Optional[str] = None
    ) -> list[dict]:
        """Prepare messages for OpenAI API"""
        messages = [{"role": "system", "content": system}]
        
        if image_data:
            # Encode image for vision
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            user_content = [
                {"type": "text", "text": user},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mime_type};base64,{encoded_image}",
                        "detail": "high"
                    }
                }
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user})
            
        return messages
    
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
        
        messages = self._prepare_messages(system, user, image_data, image_mime_type)
        
        # Add JSON format instruction
        messages[0]["content"] += f"\n\nProvide response as valid JSON matching this schema example: {schema.model_json_schema()}"
        
        try:
            if stream:
                return self._stream_completion(messages, max_output_tokens, start_time, schema)
            else:
                return self._non_stream_completion(messages, max_output_tokens, start_time, schema)
                
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit: {e}")
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI timeout: {e}")
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}")
    
    def _non_stream_completion(
        self, 
        messages: list[dict], 
        max_output_tokens: int, 
        start_time: float,
        schema: type[BaseModel]
    ) -> LLMResponse:
        """Non-streaming completion"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_output_tokens,  # Use new parameter name
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            duration = time.time() - start_time
            content = response.choices[0].message.content
            
            if not content:
                raise LLMValidationError("Empty response from OpenAI")
            
            # Parse and validate JSON
            try:
                parsed_json = orjson.loads(content)
                validated_data = schema.model_validate(parsed_json)
                result_dict = validated_data.model_dump()
            except orjson.JSONDecodeError as e:
                raise LLMValidationError(f"Invalid JSON from OpenAI: {e}", content[:500])
            except ValidationError as e:
                raise LLMValidationError(f"Schema validation failed: {e}", content[:500])
            
            # Extract usage info
            usage_info = response.usage
            input_tokens = usage_info.prompt_tokens if usage_info else None
            output_tokens = usage_info.completion_tokens if usage_info else None
            cost = self._estimate_cost(input_tokens, output_tokens)
            
            usage = LLMUsage(
                provider=self.provider_name,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                duration_seconds=duration,
                meta={"finish_reason": response.choices[0].finish_reason}
            )
            
            return LLMResponse(
                content=result_dict,
                usage=usage,
                raw_response=content
            )
            
        except Exception as e:
            if isinstance(e, (LLMValidationError, LLMError)):
                raise
            raise LLMError(f"OpenAI completion failed: {e}")
    
    def _stream_completion(
        self, 
        messages: list[dict], 
        max_output_tokens: int, 
        start_time: float,
        schema: type[BaseModel]
    ) -> Iterable[str]:
        """Streaming completion - yields text chunks"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_output_tokens,
                temperature=0.1,
                response_format={"type": "json_object"},
                stream=True
            )
            
            accumulated_content = ""
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta_content = chunk.choices[0].delta.content
                    accumulated_content += delta_content
                    yield delta_content
            
            # Final validation (though client will handle this)
            if accumulated_content:
                try:
                    parsed_json = orjson.loads(accumulated_content)
                    schema.model_validate(parsed_json)
                except (orjson.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Stream validation failed: {e}")
                    
        except openai.RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit: {e}")
        except openai.APITimeoutError as e:
            raise LLMTimeoutError(f"OpenAI timeout: {e}")
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}")
        except Exception as e:
            raise LLMError(f"OpenAI streaming failed: {e}")
