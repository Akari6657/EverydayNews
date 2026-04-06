"""Reduce-stage summarization from ClusterSummary to FinalBriefing."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from .models import AppConfig, ClusterSummary, FinalBriefing
from .prompts import (
    REDUCE_JSON_RETRY_SUFFIX,
    REDUCE_SYSTEM_PROMPT,
    REDUCE_USER_PROMPT_TEMPLATE,
)

LOGGER = logging.getLogger(__name__)


def build_final_briefing(
    summaries: list[ClusterSummary],
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
    payload, response = _request_reduce_payload(llm_client, config, prompt)
    return _parse_final_briefing(
        payload=payload,
        selected=selected,
        config=config,
        response=response,
        generated_at=generated_at,
        prior_token_usage=token_usage,
    )


def _select_summaries(summaries: list[ClusterSummary], config: AppConfig) -> list[ClusterSummary]:
    """Select the top summaries that should participate in reduce."""

    ranked = sorted(
        summaries,
        key=lambda summary: (
            -summary.importance,
            -len(summary.source_names),
            summary.cluster_id,
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
        total_clusters=0,
        total_sources=0,
        generated_at=generated_at,
        token_usage=usage,
        model=config.llm.model,
    )


def _build_summaries_payload(summaries: list[ClusterSummary]) -> str:
    """Render all selected summaries into the reduce prompt payload."""

    return "\n".join(_summary_block(summary) for summary in summaries)


def _summary_block(summary: ClusterSummary) -> str:
    """Render one map-stage summary block for the model prompt."""

    lines = [
        f"[cluster_id: {summary.cluster_id}]",
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


def _request_reduce_payload(client: Any, config: AppConfig, prompt: str) -> tuple[dict[str, Any], Any]:
    """Request reduce-stage JSON, retrying when invalid."""

    max_retries = config.summarizer.reduce.max_retries
    current_prompt = prompt
    for attempt in range(max_retries):
        try:
            response = _request_reduce(client, config, current_prompt)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise RuntimeError("Failed to request reduce-stage summary after retries") from exc
            LOGGER.warning("Reduce-stage request attempt %s failed: %s", attempt + 1, exc)
            time.sleep(2**attempt)
            continue
        raw_text = _extract_response_text(response)
        try:
            payload = json.loads(raw_text)
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


def _parse_final_briefing(
    payload: dict[str, Any],
    selected: list[ClusterSummary],
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

    summary_lookup = {summary.cluster_id: summary for summary in selected}
    topics: dict[str, list[ClusterSummary]] = {}
    seen_cluster_ids: set[str] = set()
    for topic_name, items in topic_payload.items():
        if not isinstance(topic_name, str) or not topic_name.strip():
            raise ValueError("Topic names in reduce-stage JSON must be non-empty strings")
        if not isinstance(items, list):
            raise ValueError("Each topic in reduce-stage JSON must map to a list")
        parsed_items: list[ClusterSummary] = []
        for item in items:
            parsed_summary = _parse_topic_item(item, topic_name.strip(), summary_lookup)
            parsed_items.append(parsed_summary)
            seen_cluster_ids.add(parsed_summary.cluster_id)
        topics[topic_name.strip()] = parsed_items

    _append_missing_summaries(topics, selected, seen_cluster_ids)
    usage = _merge_token_usage(prior_token_usage, _response_token_usage(response))
    return FinalBriefing(
        date=generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        overview_zh=overview.strip(),
        topics=topics,
        total_clusters=len(selected),
        total_sources=len({name for summary in selected for name in summary.source_names}),
        generated_at=generated_at,
        token_usage=usage,
        model=str(getattr(response, "model", config.llm.model)),
    )


def _parse_topic_item(
    item: Any,
    topic_name: str,
    summary_lookup: dict[str, ClusterSummary],
) -> ClusterSummary:
    """Parse one topic item, falling back to map-stage fields when needed."""

    if isinstance(item, str):
        cluster_id = item.strip()
        if not cluster_id:
            raise ValueError("Reduce-stage topic items cannot include empty cluster ids")
        base = _lookup_cluster(cluster_id, summary_lookup)
        return _with_topic(base, topic_name)
    if not isinstance(item, dict):
        raise ValueError("Reduce-stage topic items must be objects or cluster id strings")

    cluster_id = item.get("cluster_id")
    if not isinstance(cluster_id, str) or not cluster_id.strip():
        raise ValueError("Reduce-stage topic items must include a non-empty 'cluster_id'")
    base = _lookup_cluster(cluster_id.strip(), summary_lookup)

    headline = _optional_string(item.get("headline_zh")) or base.headline_zh
    summary_zh = _optional_string(item.get("summary_zh")) or base.summary_zh
    importance = _coerce_importance(item.get("importance"), base.importance)
    entities = _coerce_string_list(item.get("entities"), base.entities)
    source_names = _coerce_string_list(item.get("source_names"), base.source_names)
    primary_link = _optional_string(item.get("primary_link")) or base.primary_link

    return ClusterSummary(
        cluster_id=base.cluster_id,
        topic=topic_name,
        headline_zh=headline,
        summary_zh=summary_zh,
        importance=importance,
        entities=entities,
        source_names=source_names,
        primary_link=primary_link,
    )


def _lookup_cluster(cluster_id: str, summary_lookup: dict[str, ClusterSummary]) -> ClusterSummary:
    """Look up one cluster id from the selected summaries."""

    try:
        return summary_lookup[cluster_id]
    except KeyError as exc:
        raise ValueError(f"Unknown cluster_id returned by reduce-stage JSON: {cluster_id}") from exc


def _with_topic(summary: ClusterSummary, topic_name: str) -> ClusterSummary:
    """Return a copy of a summary with a replaced topic."""

    return ClusterSummary(
        cluster_id=summary.cluster_id,
        topic=topic_name,
        headline_zh=summary.headline_zh,
        summary_zh=summary.summary_zh,
        importance=summary.importance,
        entities=list(summary.entities),
        source_names=list(summary.source_names),
        primary_link=summary.primary_link,
    )


def _append_missing_summaries(
    topics: dict[str, list[ClusterSummary]],
    selected: list[ClusterSummary],
    seen_cluster_ids: set[str],
) -> None:
    """Keep omitted summaries by appending them under their original topic."""

    missing = [summary for summary in selected if summary.cluster_id not in seen_cluster_ids]
    if missing:
        LOGGER.warning(
            "Reduce-stage JSON omitted %s summaries; appending them with original topics",
            len(missing),
        )
    for summary in missing:
        topics.setdefault(summary.topic, []).append(summary)


def _optional_string(value: Any) -> str:
    """Return a stripped string or an empty string."""

    if isinstance(value, str):
        return value.strip()
    return ""


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
