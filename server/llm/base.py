"""
LLM client protocol and common types for surgical AI analysis.
"""
from __future__ import annotations
from typing import Protocol, Optional, Iterable, Dict, Any, Union
from pydantic import BaseModel
import time
from dataclasses import dataclass


@dataclass
class LLMUsage:
    """Token usage and cost information"""
    provider: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    duration_seconds: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from LLM with usage metadata"""
    content: Union[Dict[str, Any], str]
    usage: LLMUsage
    raw_response: Optional[str] = None


class LLMClient(Protocol):
    """Protocol for LLM clients with structured output support"""
    
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
        """
        Complete with JSON output validated against schema.
        
        Args:
            system: System prompt
            user: User prompt  
            schema: Pydantic model class for validation
            max_output_tokens: Maximum tokens to generate
            stream: If True, return iterator of text chunks
            image_data: Optional image bytes for vision models
            image_mime_type: MIME type of image (e.g., 'image/jpeg')
            
        Returns:
            LLMResponse with validated dict if stream=False
            Iterator of text chunks if stream=True
        """
        ...
    
    @property
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'openai', 'anthropic')"""
        ...
    
    @property
    def model_name(self) -> str:
        """Name of the model being used"""
        ...


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class LLMValidationError(LLMError):
    """JSON validation failed"""
    def __init__(self, message: str, raw_content: str = ""):
        super().__init__(message)
        self.raw_content = raw_content


class LLMTimeoutError(LLMError):
    """Request timed out"""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded"""
    pass
