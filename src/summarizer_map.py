"""Map-stage summarization from ArticleCluster to ClusterSummary."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import timezone
from typing import Any, Iterable, Sequence

from .models import AppConfig, ArticleCluster, ClusterSummary, MapSummariesResult
from .prompts import MAP_JSON_RETRY_SUFFIX, MAP_SYSTEM_PROMPT, MAP_USER_PROMPT_TEMPLATE

LOGGER = logging.getLogger(__name__)


def summarize_clusters(
    clusters: list[ArticleCluster],
    config: AppConfig,
    client: Any | None = None,
) -> list[ClusterSummary]:
    """Summarize deduplicated article clusters into structured summaries."""

    return summarize_clusters_with_usage(clusters, config, client=client).summaries


def summarize_clusters_with_usage(
    clusters: list[ArticleCluster],
    config: AppConfig,
    client: Any | None = None,
) -> MapSummariesResult:
    """Summarize clusters and aggregate token usage across all map batches."""

    if not clusters:
        return MapSummariesResult(
            summaries=[],
            token_usage={"input_tokens": 0, "output_tokens": 0},
            model=config.llm.model,
        )
    llm_client = client or _create_client(config)
    summaries: list[ClusterSummary] = []
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    model = config.llm.model
    for batch in _chunked(clusters, config.summarizer.map.batch_size):
        batch_summaries, batch_usage, batch_model = _summarize_batch(batch, config, llm_client)
        summaries.extend(batch_summaries)
        token_usage = _merge_token_usage(token_usage, batch_usage)
        if batch_model:
            model = batch_model
    return MapSummariesResult(summaries=summaries, token_usage=token_usage, model=model)


def _summarize_batch(
    clusters: list[ArticleCluster],
    config: AppConfig,
    client: Any,
) -> tuple[list[ClusterSummary], dict[str, int], str | None]:
    """Summarize one batch of clusters, retrying invalid JSON once."""

    prompt = MAP_USER_PROMPT_TEMPLATE.format(clusters_payload=_build_clusters_payload(clusters))
    max_retries = config.summarizer.map.max_retries
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    for attempt in range(max_retries):
        try:
            response = _request_batch(client, config, prompt)
        except Exception as exc:
            if attempt == max_retries - 1:
                LOGGER.error("Skipping map-stage batch after API failure: %s", exc)
                return [], token_usage, None
            LOGGER.warning("Map-stage request attempt %s failed: %s", attempt + 1, exc)
            time.sleep(2**attempt)
            continue

        token_usage = _merge_token_usage(token_usage, _response_token_usage(response))
        raw_text = _extract_response_text(response)
        try:
            payload = json.loads(raw_text)
            items = _extract_items(payload)
            if len(items) != len(clusters):
                raise ValueError(
                    f"Expected {len(clusters)} summaries in batch, received {len(items)}"
                )
            return (
                [
                    _parse_cluster_summary(item, cluster)
                    for item, cluster in zip(items, clusters)
                ],
                token_usage,
                str(getattr(response, "model", config.llm.model)),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            if attempt == max_retries - 1:
                LOGGER.error("Skipping map-stage batch after invalid JSON: %s", exc)
                return [], token_usage, str(getattr(response, "model", config.llm.model))
            LOGGER.warning("Invalid map-stage JSON on attempt %s: %s", attempt + 1, exc)
            prompt = prompt + MAP_JSON_RETRY_SUFFIX
            time.sleep(2**attempt)
    return [], token_usage, None


def _request_batch(client: Any, config: AppConfig, prompt: str) -> Any:
    """Send one JSON-mode batch request to DeepSeek."""

    return client.chat.completions.create(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": MAP_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


def _build_clusters_payload(clusters: Sequence[ArticleCluster]) -> str:
    """Build the prompt payload for a batch of clusters."""

    return "\n".join(_cluster_block(cluster) for cluster in clusters)


def _cluster_block(cluster: ArticleCluster) -> str:
    """Render one cluster as a prompt block."""

    published_text = cluster.primary.published.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"[cluster_id: {cluster.cluster_id}]",
        f"来源: {', '.join(cluster.source_names)} ({cluster.source_count} 家报道)",
        f"主标题: {cluster.primary.title}",
        f"主摘要: {cluster.primary.description}",
        f"发布时间: {published_text}",
        f"主链接: {cluster.primary.link}",
    ]
    if cluster.duplicates:
        lines.append("相关报道:")
        for article in cluster.duplicates:
            lines.append(f"- {article.source_name}: {article.title}")
    lines.append("---")
    return "\n".join(lines)


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    """Normalize JSON payloads into a list of summary objects."""

    if isinstance(payload, dict):
        items = payload.get("items", [])
    elif isinstance(payload, list):
        items = payload
    else:
        raise TypeError("Map-stage JSON payload must be an object or list")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise TypeError("Map-stage JSON payload must contain a list of objects")
    return items


def _parse_cluster_summary(item: dict[str, Any], cluster: ArticleCluster) -> ClusterSummary:
    """Convert one JSON object into a ClusterSummary dataclass."""

    topic = _require_non_empty_string(item, "topic")
    headline_zh = _require_non_empty_string(item, "headline_zh")
    summary_zh = _require_non_empty_string(item, "summary_zh")
    importance = _coerce_importance(item.get("importance"))
    entities = _coerce_string_list(item.get("entities", []))
    return ClusterSummary(
        cluster_id=cluster.cluster_id,
        topic=topic,
        headline_zh=headline_zh,
        summary_zh=summary_zh,
        importance=importance,
        entities=entities,
        source_names=list(cluster.source_names),
        primary_link=cluster.primary.link,
    )


def _require_non_empty_string(item: dict[str, Any], key: str) -> str:
    """Require a non-empty string field in model JSON."""

    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or empty '{key}' in map-stage JSON")
    return value.strip()


def _coerce_importance(value: Any) -> int:
    """Convert a model importance field to a bounded integer."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("'importance' must be numeric")
    return max(0, min(10, int(round(float(value)))))


def _coerce_string_list(value: Any) -> list[str]:
    """Convert model entities payloads into a list of strings."""

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _chunked(items: Sequence[ArticleCluster], size: int) -> Iterable[list[ArticleCluster]]:
    """Yield fixed-size chunks from an input sequence."""

    for index in range(0, len(items), size):
        yield list(items[index : index + size])


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


def _response_token_usage(response: Any) -> dict[str, int]:
    """Extract prompt/completion token counts from a response."""

    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _merge_token_usage(
    previous: dict[str, int],
    current: dict[str, int],
) -> dict[str, int]:
    """Merge two token usage dictionaries."""

    return {
        "input_tokens": int(previous.get("input_tokens", 0)) + int(
            current.get("input_tokens", 0)
        ),
        "output_tokens": int(previous.get("output_tokens", 0)) + int(
            current.get("output_tokens", 0)
        ),
    }
