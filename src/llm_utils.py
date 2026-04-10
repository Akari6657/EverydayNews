"""Shared helpers for OpenAI-compatible LLM calls."""

from __future__ import annotations

import json
import os
from typing import Any

from .models import AppConfig


def create_client(config: AppConfig) -> Any:
    """Create an OpenAI-compatible client for the configured provider."""

    from openai import OpenAI

    api_key = os.getenv(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{config.llm.api_key_env}' is required")
    return OpenAI(api_key=api_key, base_url=config.llm.base_url)


def extract_response_text(response: Any) -> str:
    """Extract assistant text from an OpenAI-compatible response object."""

    choices = getattr(response, "choices", [])
    if not choices:
        raise RuntimeError("LLM response did not include any choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def load_json_payload(raw_text: str) -> Any:
    """Load JSON, tolerating fenced blocks or surrounding text."""

    stripped = raw_text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    for opening, closing in (("{", "}"), ("[", "]")):
        start = stripped.find(opening)
        end = stripped.rfind(closing)
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1].strip())
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON content found", raw_text, 0)


def response_token_usage(response: Any) -> dict[str, int]:
    """Extract prompt/completion token counts from a response."""

    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def merge_token_usage(*usage_dicts: dict[str, int] | None) -> dict[str, int]:
    """Merge any number of token-usage dictionaries."""

    merged = {"input_tokens": 0, "output_tokens": 0}
    for usage in usage_dicts:
        if not usage:
            continue
        merged["input_tokens"] += int(usage.get("input_tokens", 0))
        merged["output_tokens"] += int(usage.get("output_tokens", 0))
    return merged


def is_content_risk_error(exc: Exception) -> bool:
    """Return whether a provider error indicates content-risk blocking."""

    return "Content Exists Risk" in str(exc)
