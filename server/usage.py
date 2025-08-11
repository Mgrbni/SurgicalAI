"""
Token usage and cost logging utilities.
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

from server.settings import get_cost_estimate, redact_secret

logger = logging.getLogger(__name__)

@dataclass
class LLMUsage:
    """Token usage and performance metrics"""
    request_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: float
    fallback_used: bool = False
    success: bool = True
    error: Optional[str] = None

class UsageLogger:
    """Logger for LLM token usage and costs"""
    
    def __init__(self, log_dir: str = "runs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "usage.jsonl"
    
    def log(self, usage: LLMUsage):
        """Append usage record to JSONL file"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                json.dump(asdict(usage), f)
                f.write("\n")
            
            # Console logging with redacted info
            cost_str = f"${usage.cost_usd:.4f}" if usage.cost_usd > 0 else "$0.0000"
            fallback_str = f" (fallback={usage.fallback_used})" if usage.fallback_used else ""
            print(f"PROVIDER={usage.provider} MODEL={usage.model} in={usage.input_tokens} out={usage.output_tokens} cost={cost_str} latency={usage.latency_ms/1000:.1f}s{fallback_str}")
            
        except Exception as e:
            print(f"Failed to log usage: {e}")
    
    def get_usage_summary(
        self, 
        since_timestamp: Optional[float] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get usage summary from logs"""
        if not self.log_file.exists():
            return {"total_requests": 0, "total_cost": 0.0, "total_tokens": 0}
        
        total_requests = 0
        total_cost = 0.0
        total_tokens = 0
        provider_breakdown = {}
        
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        
                        # Filter by timestamp
                        if since_timestamp and entry.get("timestamp", 0) < since_timestamp:
                            continue
                        
                        # Filter by provider
                        if provider and entry.get("provider") != provider:
                            continue
                        
                        total_requests += 1
                        total_cost += entry.get("cost_usd", 0.0) or 0.0
                        total_tokens += (entry.get("input_tokens", 0) or 0) + (entry.get("output_tokens", 0) or 0)
                        
                        # Provider breakdown
                        entry_provider = entry.get("provider", "unknown")
                        if entry_provider not in provider_breakdown:
                            provider_breakdown[entry_provider] = {
                                "requests": 0,
                                "cost": 0.0,
                                "tokens": 0
                            }
                        
                        provider_breakdown[entry_provider]["requests"] += 1
                        provider_breakdown[entry_provider]["cost"] += entry.get("cost_usd", 0.0) or 0.0
                        provider_breakdown[entry_provider]["tokens"] += (entry.get("input_tokens", 0) or 0) + (entry.get("output_tokens", 0) or 0)
                        
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to read usage logs: {e}")
        
        return {
            "total_requests": total_requests,
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "provider_breakdown": provider_breakdown
        }

# Global logger instance
_logger = UsageLogger()

def log_usage(
    request_id: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    fallback_used: bool = False,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """Log LLM usage with cost calculation"""
    cost_usd = get_cost_estimate(provider, model, input_tokens, output_tokens)
    
    usage = LLMUsage(
        request_id=request_id,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        timestamp=time.time(),
        fallback_used=fallback_used,
        success=success,
        error=redact_secret(error) if error else None
    )
    
    _logger.log(usage)

def extract_openai_usage(response) -> tuple[int, int]:
    """Extract token usage from OpenAI response"""
    try:
        if hasattr(response, 'usage') and response.usage:
            return (
                response.usage.prompt_tokens or 0,
                response.usage.completion_tokens or 0
            )
    except Exception:
        pass
    return 0, 0

def extract_anthropic_usage(response) -> tuple[int, int]:
    """Extract token usage from Anthropic response"""
    try:
        if hasattr(response, 'usage') and response.usage:
            return (
                response.usage.input_tokens or 0,
                response.usage.output_tokens or 0
            )
    except Exception:
        pass
    return 0, 0
