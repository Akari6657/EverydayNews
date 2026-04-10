"""Reduce-stage summarization from thread summaries to the final briefing."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from .models import AppConfig, FinalBriefing, ThreadSummary
from .prompts import (
    REDUCE_JSON_RETRY_SUFFIX,
    REDUCE_SAFE_USER_PROMPT_TEMPLATE,
    REDUCE_SYSTEM_PROMPT,
    REDUCE_USER_PROMPT_TEMPLATE,
)

LOGGER = logging.getLogger(__name__)


def build_final_briefing(
    summaries: list[ThreadSummary],
    config: AppConfig,
    client: Any | None = None,
    now: datetime | None = None,
    token_usage: dict[str, int] | None = None,
) -> FinalBriefing:
    """Reduce map-stage summaries into a final structured daily briefing."""

    generated_at = now or datetime.now(timezone.utc)
    selected = _select_summaries(summaries, config)
    if not selected:
        return _empty_briefing(config, generated_at, token_usage)

    llm_client = client or _create_client(config)
    prompt = REDUCE_USER_PROMPT_TEMPLATE.format(
        summaries_payload=_build_summaries_payload(selected)
    )
    safe_prompt = REDUCE_SAFE_USER_PROMPT_TEMPLATE.format(
        summaries_payload=_build_safe_summaries_payload(selected)
    )
    try:
        payload, response = _request_reduce_payload(llm_client, config, prompt, safe_prompt=safe_prompt)
        return _parse_final_briefing(
            payload=payload,
            selected=selected,
            config=config,
            response=response,
            generated_at=generated_at,
            prior_token_usage=token_usage,
        )
    except Exception as exc:
        LOGGER.warning("Reduce-stage failed; falling back to local briefing assembly: %s", exc)
        return _fallback_briefing(selected, config, generated_at, token_usage)


def count_reduce_candidates(summaries: list[ThreadSummary], config: AppConfig) -> int:
    """Return how many summaries survive reduce-stage prefiltering."""

    return len(_select_summaries(summaries, config))


def _select_summaries(summaries: list[ThreadSummary], config: AppConfig) -> list[ThreadSummary]:
    """Select the top summaries that should participate in reduce."""

    filtered = [
        summary
        for summary in summaries
        if summary.importance >= config.pipeline.importance_threshold
        and _passes_summary_filters(summary, config)
    ]
    ranked = sorted(
        filtered,
        key=lambda summary: (
            -summary.importance,
            -len(summary.source_names),
            summary.thread_id,
        ),
    )
    return ranked[: config.summarizer.reduce.top_k]


def _empty_briefing(
    config: AppConfig,
    generated_at: datetime,
    token_usage: dict[str, int] | None,
) -> FinalBriefing:
    """Return an empty structured briefing."""

    usage = _merge_token_usage(token_usage, {"input_tokens": 0, "output_tokens": 0})
    return FinalBriefing(
        date=generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        overview_zh="今日暂无新的头条新闻。",
        topics={},
        total_threads=0,
        total_sources=0,
        generated_at=generated_at,
        token_usage=usage,
        model=config.llm.model,
    )


def _build_summaries_payload(summaries: list[ThreadSummary]) -> str:
    """Render all selected summaries into the reduce prompt payload."""

    return "\n".join(_summary_block(summary) for summary in summaries)


def _build_safe_summaries_payload(summaries: list[ThreadSummary]) -> str:
    """Render a minimal payload for safer reduce retries."""

    return "\n".join(_safe_summary_block(summary) for summary in summaries)


def _summary_block(summary: ThreadSummary) -> str:
    """Render one map-stage summary block for the model prompt."""

    lines = [
        f"[thread_id: {summary.thread_id}]",
        f"主题: {summary.topic}",
        f"中文标题: {summary.headline_zh}",
        f"摘要: {summary.summary_zh}",
        f"重要性: {summary.importance}/10",
        f"来源: {', '.join(summary.source_names)}",
        f"实体: {', '.join(summary.entities) if summary.entities else '无'}",
        f"链接: {summary.primary_link}",
        "---",
    ]
    return "\n".join(lines)


def _safe_summary_block(summary: ThreadSummary) -> str:
    """Render a reduced-risk block for fallback reduce prompts."""

    lines = [
        f"[thread_id: {summary.thread_id}]",
        f"主题: {summary.topic}",
        f"中文标题: {summary.headline_zh}",
        f"重要性: {summary.importance}/10",
        f"来源数: {len(summary.source_names)}",
        "---",
    ]
    return "\n".join(lines)


def _request_reduce_payload(
    client: Any,
    config: AppConfig,
    prompt: str,
    safe_prompt: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Request reduce-stage JSON, retrying when invalid."""

    max_retries = config.summarizer.reduce.max_retries
    current_prompt = prompt
    used_safe_prompt = False
    for attempt in range(max_retries):
        try:
            response = _request_reduce(client, config, current_prompt)
        except Exception as exc:
            if safe_prompt and not used_safe_prompt and _is_content_risk_error(exc):
                LOGGER.warning("Reduce-stage hit content risk; retrying with safer prompt")
                current_prompt = safe_prompt
                used_safe_prompt = True
                time.sleep(2**attempt)
                continue
            if attempt == max_retries - 1:
                raise RuntimeError("Failed to request reduce-stage summary after retries") from exc
            LOGGER.warning("Reduce-stage request attempt %s failed: %s", attempt + 1, exc)
            time.sleep(2**attempt)
            continue
        raw_text = _extract_response_text(response)
        try:
            payload = _load_json_payload(raw_text)
            if not isinstance(payload, dict):
                raise TypeError("Reduce-stage JSON payload must be an object")
            return payload, response
        except (json.JSONDecodeError, TypeError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError("Failed to parse reduce-stage JSON after retries") from exc
            LOGGER.warning("Invalid reduce-stage JSON on attempt %s: %s", attempt + 1, exc)
            current_prompt = current_prompt + REDUCE_JSON_RETRY_SUFFIX
            time.sleep(2**attempt)
    raise RuntimeError("Reduce-stage request loop ended unexpectedly")


def _request_reduce(client: Any, config: AppConfig, prompt: str) -> Any:
    """Send one JSON-mode reduce request to DeepSeek."""

    return client.chat.completions.create(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


def _fallback_briefing(
    selected: list[ThreadSummary],
    config: AppConfig,
    generated_at: datetime,
    prior_token_usage: dict[str, int] | None,
) -> FinalBriefing:
    """Assemble a deterministic local briefing when reduce LLM calls fail."""

    topics: dict[str, list[ThreadSummary]] = {}
    for summary in selected:
        topics.setdefault(summary.topic, []).append(summary)
    for topic_name, items in list(topics.items()):
        ranked = sorted(
            items,
            key=lambda item: (-item.importance, -len(item.source_names), item.thread_id),
        )
        topics[topic_name] = ranked[: config.pipeline.max_items_per_topic]
    displayed = [summary for items in topics.values() for summary in items]
    return FinalBriefing(
        date=generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        overview_zh=_fallback_overview(topics),
        topics=topics,
        total_threads=len(displayed),
        total_sources=len({name for summary in displayed for name in summary.source_names}),
        generated_at=generated_at,
        token_usage=_merge_token_usage(prior_token_usage, {"input_tokens": 0, "output_tokens": 0}),
        model=f"{config.llm.model} (fallback)",
    )


def _fallback_overview(topics: dict[str, list[ThreadSummary]]) -> str:
    """Generate a simple Chinese overview without an extra LLM call."""

    if not topics:
        return "今日暂无新的头条新闻。"
    topic_parts = [
        f"{topic_name}（{len(items)}条）"
        for topic_name, items in sorted(
            topics.items(),
            key=lambda item: (-len(item[1]), -max(summary.importance for summary in item[1]), item[0]),
        )[:3]
    ]
    headline_parts = [
        items[0].headline_zh
        for _, items in sorted(
            topics.items(),
            key=lambda item: (-len(item[1]), -max(summary.importance for summary in item[1]), item[0]),
        )
        if items
    ][:3]
    overview = f"今日简报重点涵盖{'、'.join(topic_parts)}。"
    if headline_parts:
        overview += f"主要关注事件包括{'、'.join(headline_parts)}。"
    return overview


def _parse_final_briefing(
    payload: dict[str, Any],
    selected: list[ThreadSummary],
    config: AppConfig,
    response: Any,
    generated_at: datetime,
    prior_token_usage: dict[str, int] | None,
) -> FinalBriefing:
    """Convert reduce-stage JSON into a FinalBriefing dataclass."""

    overview = payload.get("overview_zh")
    if not isinstance(overview, str) or not overview.strip():
        raise ValueError("Reduce-stage JSON must include a non-empty 'overview_zh'")

    topic_payload = payload.get("topics", {})
    if not isinstance(topic_payload, dict):
        raise ValueError("Reduce-stage JSON must include an object 'topics'")

    summary_lookup = {summary.thread_id: summary for summary in selected}
    topics: dict[str, list[ThreadSummary]] = {}
    seen_thread_ids: set[str] = set()
    for topic_name, items in topic_payload.items():
        if not isinstance(topic_name, str) or not topic_name.strip():
            raise ValueError("Topic names in reduce-stage JSON must be non-empty strings")
        if not isinstance(items, list):
            raise ValueError("Each topic in reduce-stage JSON must map to a list")
        parsed_items: list[ThreadSummary] = []
        for item in items:
            parsed_summary = _parse_topic_item(item, topic_name.strip(), summary_lookup)
            parsed_items.append(parsed_summary)
            seen_thread_ids.add(parsed_summary.thread_id)
        topics[topic_name.strip()] = parsed_items

    _append_missing_summaries(topics, selected, seen_thread_ids)
    _trim_topics(topics, config.pipeline.max_items_per_topic)
    displayed = [summary for items in topics.values() for summary in items]
    usage = _merge_token_usage(prior_token_usage, _response_token_usage(response))
    return FinalBriefing(
        date=generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        overview_zh=overview.strip(),
        topics=topics,
        total_threads=len(displayed),
        total_sources=len({name for summary in displayed for name in summary.source_names}),
        generated_at=generated_at,
        token_usage=usage,
        model=str(getattr(response, "model", config.llm.model)),
    )


def _load_json_payload(raw_text: str) -> Any:
    """Load JSON, tolerating fenced blocks or surrounding text."""

    stripped = raw_text.strip()
    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            candidates.append("\n".join(lines[1:-1]).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
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


def _is_content_risk_error(exc: Exception) -> bool:
    """Return whether a request failure matches DeepSeek content-risk blocking."""

    return "Content Exists Risk" in str(exc)


def _parse_topic_item(
    item: Any,
    topic_name: str,
    summary_lookup: dict[str, ThreadSummary],
) -> ThreadSummary:
    """Parse one topic item, falling back to map-stage fields when needed."""

    if isinstance(item, str):
        thread_id = item.strip()
        if not thread_id:
            raise ValueError("Reduce-stage topic items cannot include empty thread ids")
        base = _lookup_thread(thread_id, summary_lookup)
        return _with_topic(base, topic_name)
    if not isinstance(item, dict):
        raise ValueError("Reduce-stage topic items must be objects or thread id strings")

    thread_id = item.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("Reduce-stage topic items must include a non-empty 'thread_id'")
    base = _lookup_thread(thread_id.strip(), summary_lookup)

    headline = _optional_string(item.get("headline_zh")) or base.headline_zh
    summary_zh = _optional_string(item.get("summary_zh")) or base.summary_zh
    importance = _coerce_importance(item.get("importance"), base.importance)
    entities = _coerce_string_list(item.get("entities"), base.entities)
    source_names = _coerce_string_list(item.get("source_names"), base.source_names)
    primary_link = _optional_string(item.get("primary_link")) or base.primary_link

    return ThreadSummary(
        thread_id=base.thread_id,
        topic=topic_name,
        headline_zh=headline,
        summary_zh=summary_zh,
        importance=importance,
        entities=entities,
        source_names=source_names,
        primary_link=primary_link,
    )


def _lookup_thread(thread_id: str, summary_lookup: dict[str, ThreadSummary]) -> ThreadSummary:
    """Look up one thread id from the selected summaries."""

    try:
        return summary_lookup[thread_id]
    except KeyError as exc:
        raise ValueError(f"Unknown thread_id returned by reduce-stage JSON: {thread_id}") from exc


def _with_topic(summary: ThreadSummary, topic_name: str) -> ThreadSummary:
    """Return a copy of a summary with a replaced topic."""

    return ThreadSummary(
        thread_id=summary.thread_id,
        topic=topic_name,
        headline_zh=summary.headline_zh,
        summary_zh=summary.summary_zh,
        importance=summary.importance,
        entities=list(summary.entities),
        source_names=list(summary.source_names),
        primary_link=summary.primary_link,
    )


def _append_missing_summaries(
    topics: dict[str, list[ThreadSummary]],
    selected: list[ThreadSummary],
    seen_thread_ids: set[str],
) -> None:
    """Keep omitted summaries by appending them under their original topic."""

    missing = [summary for summary in selected if summary.thread_id not in seen_thread_ids]
    if missing:
        LOGGER.warning(
            "Reduce-stage JSON omitted %s summaries; appending them with original topics",
            len(missing),
        )
    for summary in missing:
        topics.setdefault(summary.topic, []).append(summary)


def _trim_topics(topics: dict[str, list[ThreadSummary]], limit: int) -> None:
    """Trim each topic section to a maximum number of items."""

    for topic_name, items in list(topics.items()):
        topics[topic_name] = items[:limit]


def _optional_string(value: Any) -> str:
    """Return a stripped string or an empty string."""

    if isinstance(value, str):
        return value.strip()
    return ""


def _passes_summary_filters(summary: ThreadSummary, config: AppConfig) -> bool:
    """Return whether a summary should remain eligible for reduce."""

    haystack = _normalize_filter_text(
        f"{summary.headline_zh}\n{summary.summary_zh}\n{summary.primary_link}"
    )
    for keyword in config.pipeline.exclude_summary_keywords:
        normalized = _normalize_filter_text(keyword)
        if normalized and normalized in haystack:
            return False
    return True


def _normalize_filter_text(text: str) -> str:
    """Normalize text for resilient summary keyword matching."""

    sanitized = text.casefold().replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", sanitized).split())


def _coerce_importance(value: Any, fallback: int) -> int:
    """Convert a reduce-stage importance field into an integer."""

    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Reduce-stage 'importance' must be numeric")
    return max(0, min(10, int(round(float(value)))))


def _coerce_string_list(value: Any, fallback: list[str]) -> list[str]:
    """Convert a JSON field into a non-empty string list."""

    if value is None:
        return list(fallback)
    if not isinstance(value, list):
        raise ValueError("Reduce-stage list fields must be arrays when provided")
    values = [str(item).strip() for item in value if str(item).strip()]
    return values or list(fallback)


def _response_token_usage(response: Any) -> dict[str, int]:
    """Extract prompt/completion token counts from a response."""

    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _merge_token_usage(
    prior_token_usage: dict[str, int] | None,
    current_token_usage: dict[str, int],
) -> dict[str, int]:
    """Merge reduce-stage token usage with any upstream usage."""

    previous = prior_token_usage or {}
    return {
        "input_tokens": int(previous.get("input_tokens", 0)) + int(
            current_token_usage.get("input_tokens", 0)
        ),
        "output_tokens": int(previous.get("output_tokens", 0)) + int(
            current_token_usage.get("output_tokens", 0)
        ),
    }


def _create_client(config: AppConfig) -> Any:
    """Create an OpenAI-compatible client for DeepSeek."""

    from openai import OpenAI

    api_key = os.getenv(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{config.llm.api_key_env}' is required")
    return OpenAI(api_key=api_key, base_url=config.llm.base_url)


def _extract_response_text(response: Any) -> str:
    """Extract the first text response from an SDK payload."""

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
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()
