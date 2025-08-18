"""
LLM client abstraction and utilities for the IUS summarization system.

This module provides synchronous OpenAI API integration with comprehensive
cost tracking and model parameter handling.
"""

import os
import time
from typing import Any


try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI package not installed. Run: pip install openai") from None

try:
    import tiktoken
except ImportError:
    raise ImportError("tiktoken package not installed. Run: pip install tiktoken") from None

from .exceptions import ValidationError
from .logging_config import get_logger


logger = get_logger(__name__)


def call_llm(text: str, model: str = "gpt-4.1-mini", system_and_user_prompt: dict[str, str] = None,
             template_vars: dict[str, str] = None, ask_user_confirmation: bool = False, **kwargs) -> dict[str, Any]:
    """
    Call LLM with comprehensive tracking and optional user confirmation.

    Args:
        text: Input text to process
        model: Model name (default: gpt-4.1-mini)
        system_and_user_prompt: Dict with "system" and "user" prompt content
        template_vars: Dict of variables for prompt template substitution
        ask_user_confirmation: Whether to ask user confirmation before expensive calls
        **kwargs: Additional model parameters

    Returns:
        Dict with response, metadata, and usage statistics
    """
    # Handle OpenAI models (gpt, o1, o3, o4, etc.)
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return _call_openai(text, model, system_and_user_prompt, template_vars, ask_user_confirmation, **kwargs)
    else:
        raise NotImplementedError(f"Local model {model} not implemented yet")


def _call_openai(text: str, model: str, system_and_user_prompt: dict[str, str] = None,
                template_vars: dict[str, str] = None, ask_user_confirmation: bool = False, 
                temperature: float = 0.0, max_completion_tokens: int = 5000, **kwargs) -> dict[str, Any]:
    """
    Call OpenAI API with pre-call cost estimation and usage tracking (synchronous).
    Handles different model parameter requirements automatically.
    
    Args:
        text: Input text to process
        model: Model name
        system_and_user_prompt: Dict with "system" and "user" prompt template strings -- NOT filled in content!
        template_vars: Dict of variables for prompt template substitution
        ask_user_confirmation: Whether to ask user confirmation before API calls
        temperature: Model temperature parameter
        max_completion_tokens: Maximum output tokens
        **kwargs: Additional model parameters

    Returns:
        {
            "response": str,
            "usage": {
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
                "input_cost": float,
                "output_cost": float,
                "total_cost": float
            },
            "model": str,
            "processing_time": float
        }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValidationError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Build messages from prompts using provided template_vars
    messages = _build_messages_from_prompts(template_vars, system_and_user_prompt)

    # Print system and user prompts for transparency
    print(f"\n📝 System Prompt:")
    system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
    if system_content:
        truncated_system = system_content[:200] + "..." if len(system_content) > 200 else system_content
        print(f"   {truncated_system}")
    else:
        print("   (no system prompt)")

    print(f"\n📝 User Prompt:")
    user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
    if user_content:
        truncated_user = user_content[:500] + "..." if len(user_content) > 500 else user_content
        print(f"   {truncated_user}")
    else:
        print("   (no user prompt)")

    # Estimate cost before making the call
    estimated_input_tokens = _estimate_token_count(messages)
    estimated_output_tokens = max_completion_tokens if _model_supports_max_completion_tokens(model) else 1000
    estimated_cost = _estimate_openai_cost(model, estimated_input_tokens, estimated_output_tokens)

    # Always print cost estimate
    print(f"\n💰 Estimated Cost: ${estimated_cost:.6f}")
    print(f"   Estimated input tokens: {estimated_input_tokens}")
    print(f"   Estimated output tokens: {estimated_output_tokens}")
    print(f"   Model: {model}")

    # Ask for user confirmation if requested
    if ask_user_confirmation:
        response = input("Proceed with API call? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            raise ValidationError("User cancelled API call")

    # Build request parameters based on model capabilities
    request_params = {
        "model": model,
        "messages": messages
    }

    # Add optional parameters only for models that support them
    if _model_supports_temperature(model):
        request_params["temperature"] = temperature

    if _model_supports_max_completion_tokens(model):
        request_params["max_completion_tokens"] = max_completion_tokens

    # Add any additional kwargs directly (let OpenAI handle unsupported parameters)
    request_params.update(kwargs)

    start_time = time.time()

    try:
        response = client.chat.completions.create(**request_params)

        processing_time = time.time() - start_time

        # Extract usage information and calculate detailed costs
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Calculate detailed cost breakdown
        input_cost = (input_tokens / 1_000_000) * _estimate_input_cost_per_1m(model)
        output_cost = (output_tokens / 1_000_000) * _estimate_output_cost_per_1m(model)
        total_cost = input_cost + output_cost

        assert total_tokens == (input_tokens + output_tokens), (
            f"Token count mismatch: total_tokens={total_tokens}, "
            f"input_tokens={input_tokens}, output_tokens={output_tokens}"
        )

        # Print detailed cost breakdown to console
        print(f"\n💰 Actual API Cost: ${total_cost:.6f}")
        print(f"   Input tokens: {input_tokens} (${input_cost:.6f})")
        print(f"   Output tokens: {output_tokens} (${output_cost:.6f})")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Model: {model}")

        # Log detailed cost breakdown to file
        _log_cost_to_file(total_cost, model, input_tokens, output_tokens, total_tokens,
                         input_cost, output_cost)

        # Capture the final prompts that were actually sent to the LLM
        final_prompts = {}
        for message in messages:
            if message["role"] == "system":
                final_prompts["system"] = message["content"]
            elif message["role"] == "user":
                final_prompts["user"] = message["content"]

        return {
            "response": response.choices[0].message.content.strip(),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6)
            },
            "model": model,
            "processing_time": processing_time,
            "finish_reason": response.choices[0].finish_reason,
            "final_prompts_used": final_prompts  # Prompts with template variables replaced
        }

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def _replace_template_vars(template: str, variables: dict[str, str]) -> str:
    """Replace template variables in the format {variable_name} with actual values."""
    result = template
    for var_name, var_value in variables.items():
        placeholder = f"{{{var_name}}}"
        result = result.replace(placeholder, var_value)
    return result


def _build_messages_from_prompts(template_vars: dict[str, str], system_and_user_prompt: dict[str, str] = None) -> list[dict[str, str]]:
    """Build messages list from system and user prompts with template variable substitution."""
    if not system_and_user_prompt:
        if not template_vars:
            raise ValueError("Both system_and_user_prompt and template_vars cannot be None when using default prompts")
        # Default prompts - use text if available, otherwise empty
        text = template_vars.get("text", "")
        return [
            {"role": "system", "content": "You are a helpful assistant that analyzes documents and provides comprehensive summaries."},
            {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
        ]

    if not template_vars:
        raise ValueError("template_vars cannot be None when using custom system_and_user_prompt")

    if "system" not in system_and_user_prompt:
        raise ValueError("Missing 'system' key in system_and_user_prompt")
    if "user" not in system_and_user_prompt:
        raise ValueError("Missing 'user' key in system_and_user_prompt")

    return [
        {"role": "system", "content": _replace_template_vars(system_and_user_prompt["system"], template_vars)},
        {"role": "user", "content": _replace_template_vars(system_and_user_prompt["user"], template_vars)}
    ]


def _estimate_token_count(messages: list[dict[str, str]]) -> int:
    """Rough estimate of token count for messages (~1.3 tokens per word)."""
    total_words = sum(len(msg["content"].split()) for msg in messages)
    return int(total_words * 1.3)  # More accurate approximation


def _estimate_input_cost_per_1m(model: str) -> float:
    """Estimate input cost per 1M tokens for model (for pre-call estimation)."""
    pricing = {
        "gpt-4.1-mini": 0.40,   # $0.40 per 1M input tokens
        "gpt-4o": 2.50,         # $2.50 per 1M input tokens
        "o1-mini": 1.10,        # $1.10 per 1M input tokens
        "o3": 2.00,        # $2.00 per 1M input tokens
        "gpt-5": 1.25,     # $1.25 per 1M input tokens
        "gpt-4.1": 2.00,   # $2.00 per 1M input tokens
        "gpt-5-mini": 0.25  # $0.25 per 1M input tokens
    }
    if model not in pricing:
        raise ValueError(f"No input cost pricing data available for model: {model}")
    return pricing[model]


def _estimate_output_cost_per_1m(model: str) -> float:
    """Estimate output cost per 1M tokens for model (for pre-call estimation)."""
    pricing = {
        "gpt-4.1-mini": 1.60,   # $1.60 per 1M output tokens
        "gpt-4o": 10.00,        # $10.00 per 1M output tokens
        "o1-mini": 4.40,        # $4.40 per 1M output tokens
        "o3": 8.00,       # $8.00 per 1M output tokens
        "gpt-5": 10.00,   # $10.00 per 1M output tokens
        "gpt-4.1": 8.00,   # $8.00 per 1M output tokens
        "gpt-5-mini": 2.00 # $2.00 per 1M output tokens
    }
    if model not in pricing:
        raise ValueError(f"No output cost pricing data available for model: {model}")
    return pricing[model]


def _estimate_openai_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost based on OpenAI pricing (as of Aug 2025). NOT 2024, 2025!
    Used for pre-call estimation.
    Raises ValueError if model pricing is not available.
    """
    input_cost_per_1m = _estimate_input_cost_per_1m(model)
    output_cost_per_1m = _estimate_output_cost_per_1m(model)

    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    total_cost = input_cost + output_cost
    return round(total_cost, 6)


def _model_supports_temperature(model: str) -> bool:
    """Check if model supports temperature parameter."""
    # o1, o3, o4 models don't support temperature
    return not model.startswith(("o1", "o3", "o4"))


def _model_supports_max_completion_tokens(model: str) -> bool:
    """Check if model supports max_completion_tokens parameter."""
    # Most other models support max_completion_tokens
    return True


def _log_cost_to_file(total_cost: float, model: str, input_tokens: int, output_tokens: int,
                     total_tokens: int, input_cost: float, output_cost: float) -> None:
    """Log detailed cost breakdown to append-only spending log file."""
    from datetime import datetime
    
    filename = "cumulative-openai-spending.txt"
    timestamp = datetime.now().isoformat()
    
    # Simple CSV format - atomic append operation, no race conditions possible
    log_line = f"{timestamp},{model},{input_tokens},{output_tokens},{total_tokens},{input_cost:.6f},{output_cost:.6f},{total_cost:.6f}\n"
    
    with open(filename, 'a') as f:
        f.write(log_line)


def calculate_total_spending(filename: str = "cumulative-openai-spending.txt") -> float:
    """Calculate total spending from the append-only spending log file."""
    total = 0.0
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split(',')
                    if len(parts) >= 8:  # Ensure we have all columns
                        total += float(parts[7])  # Last column is total_cost
    except FileNotFoundError:
        # File doesn't exist yet - no spending recorded
        return 0.0
    except (ValueError, IndexError) as e:
        print(f"Warning: Error parsing spending log line: {e}")
        # Continue processing other lines
    
    return total



