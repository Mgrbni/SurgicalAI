"""
LLM router with provider fallback and retry logic.
"""
import logging
from typing import Dict, Any, Iterable, Optional, Union
import time

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel

from .base import (
    LLMClient, LLMResponse, LLMUsage, LLMError, LLMValidationError, 
    LLMTimeoutError, LLMRateLimitError
)
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)


class LLMRouter:
    """Router with primary/secondary provider fallback"""
    
    def __init__(
        self,
        primary_client: LLMClient,
        secondary_client: Optional[LLMClient] = None,
        max_retries: int = 3
    ):
        self.primary_client = primary_client
        self.secondary_client = secondary_client
        self.max_retries = max_retries
        
    @classmethod
    def from_config(
        cls,
        provider: str,
        openai_config: Optional[Dict[str, Any]] = None,
        anthropic_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> "LLMRouter":
        """Create router from configuration"""
        
        clients = {}
        
        # Create available clients
        if openai_config and openai_config.get("api_key"):
            clients["openai"] = OpenAIClient(**openai_config)
            
        if anthropic_config and anthropic_config.get("api_key"):
            clients["anthropic"] = AnthropicClient(**anthropic_config)
        
        if not clients:
            raise ValueError("At least one provider must be configured")
        
        # Set primary and secondary
        if provider in clients:
            primary = clients[provider]
            # Use the other client as secondary
            secondary = next((c for name, c in clients.items() if name != provider), None)
        else:
            # Fallback to first available
            primary = next(iter(clients.values()))
            secondary = None
            logger.warning(f"Requested provider '{provider}' not available, using {primary.provider_name}")
        
        return cls(primary, secondary, max_retries)
    
    def complete_with_fallback(
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
        """Complete with automatic fallback to secondary provider"""
        
        # Try primary provider with retries
        try:
            return self._complete_with_retries(
                self.primary_client,
                system=system,
                user=user,
                schema=schema,
                max_output_tokens=max_output_tokens,
                stream=stream,
                image_data=image_data,
                image_mime_type=image_mime_type
            )
        except (LLMError, LLMValidationError) as e:
            logger.warning(f"Primary provider ({self.primary_client.provider_name}) failed: {e}")
            
            # Try secondary provider if available
            if self.secondary_client:
                logger.info(f"Falling back to secondary provider: {self.secondary_client.provider_name}")
                try:
                    return self._complete_with_retries(
                        self.secondary_client,
                        system=system,
                        user=user,
                        schema=schema,
                        max_output_tokens=max_output_tokens,
                        stream=stream,
                        image_data=image_data,
                        image_mime_type=image_mime_type,
                        max_attempts=1  # Only one attempt for fallback
                    )
                except (LLMError, LLMValidationError) as fallback_error:
                    logger.error(f"Secondary provider also failed: {fallback_error}")
                    raise LLMError(f"Both providers failed. Primary: {e}, Secondary: {fallback_error}")
            else:
                logger.error("No secondary provider available for fallback")
                raise
    
    def _complete_with_retries(
        self,
        client: LLMClient,
        max_attempts: Optional[int] = None,
        **kwargs
    ) -> Union[LLMResponse, Iterable[str]]:
        """Complete with retries for a specific client"""
        attempts = max_attempts or self.max_retries
        
        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
            reraise=True
        )
        def _do_complete():
            return client.complete_json(**kwargs)
            
        try:
            return _do_complete()
        except LLMValidationError as e:
            # Try to recover from validation errors
            return self._recover_from_validation_error(client, e, **kwargs)
    
    def _recover_from_validation_error(
        self,
        client: LLMClient,
        error: LLMValidationError,
        **kwargs
    ) -> Union[LLMResponse, Iterable[str]]:
        """Attempt to recover from validation errors"""
        logger.info(f"Attempting recovery from validation error: {error}")
        
        # Modify the user prompt to request a fix
        original_user = kwargs["user"]
        recovery_user = f"""Your previous output was invalid. Fix to valid JSON only.

Original request:
{original_user}

Error: {str(error)}
Raw output (first 500 chars): {getattr(error, 'raw_content', '')[:500]}

Respond with ONLY valid JSON matching the required schema."""
        
        kwargs["user"] = recovery_user
        
        try:
            # Single recovery attempt
            return client.complete_json(**kwargs)
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            raise LLMValidationError(f"Could not recover from validation error. Original: {error}, Recovery: {recovery_error}")
    
    @property
    def active_provider(self) -> str:
        """Get the name of the active primary provider"""
        return self.primary_client.provider_name
    
    @property
    def active_model(self) -> str:
        """Get the name of the active primary model"""
        return self.primary_client.model_name
