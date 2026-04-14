"""Reduce-stage summarization from thread summaries to the final briefing."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from .llm_utils import (
    create_client,
    extract_response_text,
    is_content_risk_error,
    load_json_payload,
    merge_token_usage,
    response_token_usage,
)
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

    llm_client = client or create_client(config)
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
            -summary.effective_source_count,
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

    usage = merge_token_usage(token_usage, {"input_tokens": 0, "output_tokens": 0})
    return FinalBriefing(
        date=generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        overview_zh="今日暂无新的头条新闻。",
        top_stories=[],
        other_stories=[],
        total_threads=0,
        total_sources=0,
        total_articles=0,
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
        f"来源数: {summary.effective_source_count}",
        f"来源: {', '.join(summary.source_names)}",
        f"文章数: {summary.article_count}",
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
        f"来源数: {summary.effective_source_count}",
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
            if safe_prompt and not used_safe_prompt and is_content_risk_error(exc):
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
        raw_text = extract_response_text(response)
        try:
            payload = load_json_payload(raw_text)
            if not isinstance(payload, dict):
                raise TypeError("Reduce-stage JSON payload must be an object")
            return payload, response
        except (TypeError, ValueError) as exc:
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

    top_stories, other_stories = _partition_stories(selected)
    return _build_briefing(
        overview_zh=_fallback_overview(top_stories, other_stories),
        top_stories=top_stories,
        other_stories=other_stories,
        generated_at=generated_at,
        token_usage=merge_token_usage(prior_token_usage, {"input_tokens": 0, "output_tokens": 0}),
        model=f"{config.llm.model} (fallback)",
    )


def _fallback_overview(
    top_stories: list[ThreadSummary],
    other_stories: list[ThreadSummary],
) -> str:
    """Generate a simple Chinese overview without an extra LLM call."""

    stories = top_stories + other_stories
    if not stories:
        return "今日暂无新的头条新闻。"
    lead_stories = top_stories or stories[:3]
    topic_parts = [story.topic for story in lead_stories[:3]]
    headline_parts = [story.headline_zh for story in lead_stories[:3]]
    overview = f"今日简报重点涵盖{'、'.join(topic_parts)}等主题。"
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
    top_stories, other_stories = _partition_stories(selected)
    return _build_briefing(
        overview_zh=overview.strip(),
        top_stories=top_stories,
        other_stories=other_stories,
        generated_at=generated_at,
        token_usage=merge_token_usage(prior_token_usage, response_token_usage(response)),
        model=str(getattr(response, "model", config.llm.model)),
    )


def _partition_stories(
    summaries: list[ThreadSummary],
) -> tuple[list[ThreadSummary], list[ThreadSummary]]:
    """Split summaries into top stories and secondary stories."""

    top_stories: list[ThreadSummary] = []
    other_stories: list[ThreadSummary] = []
    for summary in summaries:
        (top_stories if _is_top_story(summary) else other_stories).append(summary)
    return top_stories, other_stories


def _is_top_story(summary: ThreadSummary) -> bool:
    """Return whether a thread summary belongs in the top-stories section."""

    return summary.importance >= 6 or summary.effective_source_count >= 2


def _build_briefing(
    overview_zh: str,
    top_stories: list[ThreadSummary],
    other_stories: list[ThreadSummary],
    generated_at: datetime,
    token_usage: dict[str, int],
    model: str,
) -> FinalBriefing:
    """Build a two-layer final briefing."""

    displayed = top_stories + other_stories
    return FinalBriefing(
        date=generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        overview_zh=overview_zh,
        top_stories=top_stories,
        other_stories=other_stories,
        total_threads=len(displayed),
        total_sources=len({name for summary in displayed for name in summary.source_names}),
        total_articles=sum(summary.article_count for summary in displayed),
        generated_at=generated_at,
        token_usage=token_usage,
        model=model,
    )


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
