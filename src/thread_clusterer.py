"""LLM-based story-thread clustering for fetched RSS articles."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Sequence

from .models import AppConfig, Article, StoryThread
from .prompts import (
    THREAD_CLUSTERING_JSON_RETRY_SUFFIX,
    THREAD_CLUSTERING_PROMPT_TEMPLATE,
    THREAD_REFINEMENT_PROMPT_TEMPLATE,
    THREAD_CLUSTERING_SYSTEM_PROMPT,
)

LOGGER = logging.getLogger(__name__)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def cluster_into_threads(
    articles: list[Article],
    config: AppConfig,
    client: Any | None = None,
    now: datetime | None = None,
) -> list[StoryThread]:
    """Group fetched articles into story threads via one LLM clustering pass."""

    if not articles:
        return []
    wrapper_threads, clusterable_articles = _extract_wrapper_threads(articles, config)
    if not clusterable_articles:
        return _renumber_threads(_sort_threads(wrapper_threads))
    if not config.thread_clustering.enabled:
        LOGGER.warning("Thread clustering disabled in config; using one-article-per-thread fallback")
        threads = _one_per_thread_fallback(clusterable_articles, config)
        return _renumber_threads(_sort_threads(threads + wrapper_threads))
    if len(clusterable_articles) > config.thread_clustering.max_articles_per_call:
        threads = _cluster_large_article_set(clusterable_articles, config, client=client, now=now)
        refined = _refine_threads(threads, config, client or _create_client(config))
        return _renumber_threads(_sort_threads(refined + wrapper_threads))
    llm_client = client or _create_client(config)
    raw_threads = _request_threads_payload(
        llm_client,
        config,
        THREAD_CLUSTERING_PROMPT_TEMPLATE.format(
            articles_payload=_build_articles_payload(clusterable_articles)
        ),
        stage_label="Thread clustering",
    )
    if raw_threads is None:
        threads = _one_per_thread_fallback(clusterable_articles, config)
        return _renumber_threads(_sort_threads(threads + wrapper_threads))
    threads = _build_story_threads(raw_threads, clusterable_articles, config)
    refined_threads = _refine_threads(threads, config, llm_client)
    return _renumber_threads(_sort_threads(refined_threads + wrapper_threads))


def build_one_article_threads(
    articles: list[Article],
    config: AppConfig,
) -> list[StoryThread]:
    """Return one story thread per article for debugging or fallback use."""

    return _one_per_thread_fallback(articles, config)


def _cluster_large_article_set(
    articles: list[Article],
    config: AppConfig,
    client: Any | None,
    now: datetime | None,
) -> list[StoryThread]:
    """Cluster large article sets by chunking and renumbering the results."""

    max_per_call = config.thread_clustering.max_articles_per_call
    LOGGER.warning(
        "Thread clustering received %s articles; clustering in %s-sized chunks without merge pass",
        len(articles),
        max_per_call,
    )
    threads: list[StoryThread] = []
    for chunk in _chunked(articles, max_per_call):
        threads.extend(cluster_into_threads(chunk, config, client=client, now=now))
    return _renumber_threads(_sort_threads(threads))


def _request_threads_payload(
    client: Any,
    config: AppConfig,
    prompt: str,
    stage_label: str,
) -> list[dict[str, Any]] | None:
    """Request thread assignments from the LLM, retrying invalid JSON once."""

    for attempt in range(config.thread_clustering.max_retries):
        try:
            response = _request_threads(client, config, prompt)
        except Exception as exc:
            if attempt == config.thread_clustering.max_retries - 1:
                LOGGER.error("%s failed after API retries: %s", stage_label, exc)
                return None
            LOGGER.warning("%s request attempt %s failed: %s", stage_label, attempt + 1, exc)
            time.sleep(2**attempt)
            continue

        raw_text = _extract_response_text(response)
        try:
            payload = _load_json_payload(raw_text)
            return _extract_threads(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if attempt == config.thread_clustering.max_retries - 1:
                LOGGER.error("%s failed after invalid JSON retries: %s", stage_label, exc)
                return None
            LOGGER.warning("Invalid %s JSON on attempt %s: %s", stage_label.lower(), attempt + 1, exc)
            prompt = prompt + THREAD_CLUSTERING_JSON_RETRY_SUFFIX
            time.sleep(2**attempt)
    return None


def _request_threads(client: Any, config: AppConfig, prompt: str) -> Any:
    """Send one JSON-mode thread clustering request."""

    return client.chat.completions.create(
        model=config.thread_clustering.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": THREAD_CLUSTERING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


def _build_articles_payload(articles: Sequence[Article]) -> str:
    """Render all article headlines into the clustering prompt payload."""

    return "\n".join(
        f"[{index}] {article.source_name}/{article.category} | {article.title}"
        for index, article in enumerate(articles, start=1)
    )


def _extract_threads(payload: Any) -> list[dict[str, Any]]:
    """Validate the thread clustering response payload."""

    if not isinstance(payload, dict):
        raise TypeError("Thread clustering payload must be an object")
    threads = payload.get("threads", [])
    if not isinstance(threads, list) or not all(isinstance(item, dict) for item in threads):
        raise TypeError("Thread clustering payload must contain a 'threads' list of objects")
    return threads


def _build_story_threads(
    raw_threads: list[dict[str, Any]],
    articles: list[Article],
    config: AppConfig,
) -> list[StoryThread]:
    """Normalize raw LLM assignments into validated StoryThread objects."""

    articles_by_id = {index: article for index, article in enumerate(articles, start=1)}
    assigned_ids: set[int] = set()
    normalized: list[tuple[str, str, str, list[int]]] = []
    for raw_thread in raw_threads:
        article_ids = _normalize_article_ids(raw_thread.get("article_ids"), len(articles), assigned_ids)
        if not article_ids:
            continue
        normalized.append(
            (
                _coerce_label(raw_thread.get("topic"), "其他"),
                _coerce_label(raw_thread.get("topic_en"), "Other"),
                _coerce_label(raw_thread.get("rationale"), "模型未提供分组说明"),
                article_ids,
            )
        )
    orphaned_ids = [article_id for article_id in range(1, len(articles) + 1) if article_id not in assigned_ids]
    if orphaned_ids:
        LOGGER.warning(
            "Thread clustering produced %s orphaned article(s); keeping them as singleton threads",
            len(orphaned_ids),
        )
        for orphaned_id in orphaned_ids:
            orphaned_article = articles_by_id[orphaned_id]
            normalized.append(
                (
                    _fallback_topic(orphaned_article.title),
                    orphaned_article.title[:40].strip() or "Other",
                    "模型遗漏，已自动单独保留",
                    [orphaned_id],
                )
            )
    threads = [
        _make_story_thread(
            topic=topic,
            topic_en=topic_en,
            rationale=rationale,
            articles=[articles_by_id[article_id] for article_id in article_ids],
            config=config,
        )
        for topic, topic_en, rationale, article_ids in normalized
    ]
    return _renumber_threads(_sort_threads(_post_process_threads(threads, config)))


def _normalize_article_ids(
    value: Any,
    max_article_id: int,
    assigned_ids: set[int],
) -> list[int]:
    """Keep valid, unique article IDs and log duplicate or invalid assignments."""

    if not isinstance(value, list):
        return []
    normalized: list[int] = []
    for raw_id in value:
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            continue
        if raw_id < 1 or raw_id > max_article_id:
            LOGGER.warning("Thread clustering returned invalid article id %s; ignoring it", raw_id)
            continue
        if raw_id in assigned_ids:
            LOGGER.warning("Article id %s was assigned to multiple threads; keeping the first", raw_id)
            continue
        assigned_ids.add(raw_id)
        normalized.append(raw_id)
    return normalized


def _coerce_label(value: Any, default: str) -> str:
    """Return a trimmed non-empty string label with a safe default."""

    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _make_story_thread(
    topic: str,
    topic_en: str,
    rationale: str,
    articles: list[Article],
    config: AppConfig,
) -> StoryThread:
    """Construct one StoryThread with derived source metadata."""

    priorities = config.source_priorities()
    ordered_articles = sorted(
        articles,
        key=lambda article: (
            priorities.get(article.source_slug, len(priorities)),
            -article.published.timestamp(),
            article.source_name,
        ),
    )
    distinct_sources: list[str] = []
    for article in ordered_articles:
        if article.source_name not in distinct_sources:
            distinct_sources.append(article.source_name)
    latest_published = max(article.published for article in ordered_articles)
    return StoryThread(
        thread_id=0,
        topic=topic,
        topic_en=topic_en,
        articles=ordered_articles,
        source_names=distinct_sources,
        source_count=len(distinct_sources),
        primary=ordered_articles[0],
        latest_published=latest_published,
        rationale=rationale,
    )


def _sort_threads(threads: list[StoryThread]) -> list[StoryThread]:
    """Sort threads so multi-source and recent threads appear first."""

    return sorted(
        threads,
        key=lambda thread: (
            -thread.source_count,
            -thread.latest_published.timestamp(),
            thread.topic,
        ),
    )


def _renumber_threads(threads: list[StoryThread]) -> list[StoryThread]:
    """Renumber threads sequentially after sorting or fallback assembly."""

    return [
        StoryThread(
            thread_id=index,
            topic=thread.topic,
            topic_en=thread.topic_en,
            articles=thread.articles,
            source_names=thread.source_names,
            source_count=thread.source_count,
            primary=thread.primary,
            latest_published=thread.latest_published,
            rationale=thread.rationale,
        )
        for index, thread in enumerate(threads, start=1)
    ]


def _one_per_thread_fallback(
    articles: list[Article],
    config: AppConfig,
) -> list[StoryThread]:
    """Create one thread per article when LLM clustering is unavailable."""

    threads = [
        _make_story_thread(
            topic=_fallback_topic(article.title),
            topic_en=article.title[:40].strip(),
            rationale="Fallback: one article per thread",
            articles=[article],
            config=config,
        )
        for article in articles
    ]
    return _renumber_threads(_sort_threads(threads))


def _fallback_topic(title: str) -> str:
    """Build a short Chinese-compatible fallback topic label from a title."""

    compact = " ".join(title.split()).strip()
    return compact[:15] or "其他"


def _extract_wrapper_threads(
    articles: list[Article],
    config: AppConfig,
) -> tuple[list[StoryThread], list[Article]]:
    """Isolate generic wrapper headlines that should not steer clustering."""

    wrapper_articles: list[Article] = []
    clusterable_articles: list[Article] = []
    for article in articles:
        if _is_generic_wrapper_title(article.title):
            wrapper_articles.append(article)
        else:
            clusterable_articles.append(article)
    wrapper_threads = [
        _make_story_thread(
            topic=_fallback_topic(article.title),
            topic_en=article.title[:40].strip(),
            rationale="通用包装标题，单独保留以避免误导聚类",
            articles=[article],
            config=config,
        )
        for article in wrapper_articles
    ]
    if wrapper_articles:
        LOGGER.info(
            "Thread clustering isolated %s generic wrapper article(s) as singleton threads",
            len(wrapper_articles),
        )
    return wrapper_threads, clusterable_articles


def _is_generic_wrapper_title(title: str) -> bool:
    """Return whether a title is too generic to guide cross-article clustering."""

    normalized = _normalize_title(title)
    return (
        normalized.startswith("watch ")
        or normalized.startswith("watch")
        or normalized.startswith("morning news brief")
        or normalized.startswith("what to know")
        or normalized.startswith("what we know")
    )


def _refine_threads(
    threads: list[StoryThread],
    config: AppConfig,
    client: Any,
) -> list[StoryThread]:
    """Split oversized threads into tighter subthreads."""

    refined = list(threads)
    for _ in range(config.thread_clustering.max_refinement_rounds):
        updated: list[StoryThread] = []
        changed = False
        for thread in refined:
            if len(thread.articles) <= config.thread_clustering.max_articles_per_thread:
                updated.append(thread)
                continue
            LOGGER.info(
                "Refining oversized thread '%s' with %s articles",
                thread.topic,
                len(thread.articles),
            )
            split_threads = _split_broad_thread(thread, config, client)
            if len(split_threads) > 1:
                updated.extend(split_threads)
                changed = True
            else:
                updated.append(thread)
        refined = _sort_threads(updated)
        if not changed:
            break
    return refined


def _split_broad_thread(
    thread: StoryThread,
    config: AppConfig,
    client: Any,
) -> list[StoryThread]:
    """Ask the LLM to break one oversized thread into tighter subthreads."""

    prompt = THREAD_REFINEMENT_PROMPT_TEMPLATE.format(
        topic=thread.topic,
        topic_en=thread.topic_en,
        articles_payload=_build_articles_payload(thread.articles),
    )
    raw_threads = _request_threads_payload(
        client,
        config,
        prompt,
        stage_label=f"Thread refinement for {thread.topic}",
    )
    if raw_threads is None:
        return [thread]
    subthreads = _build_story_threads(raw_threads, thread.articles, config)
    if len(subthreads) <= 1:
        LOGGER.warning(
            "Thread refinement for '%s' did not produce meaningful splits; keeping the original thread",
            thread.topic,
        )
        return [thread]
    return subthreads


def _normalize_title(title: str) -> str:
    """Normalize titles for lightweight wrapper-pattern matching."""

    lowered = title.casefold()
    compact = NON_ALNUM_RE.sub(" ", lowered)
    return " ".join(compact.split())


def _post_process_threads(
    threads: list[StoryThread],
    config: AppConfig,
) -> list[StoryThread]:
    """Apply conservative cleanup to weakly-related thread assignments."""

    cleaned: list[StoryThread] = []
    for thread in threads:
        split_threads = _split_weak_single_source_thread(thread, config)
        cleaned.extend(split_threads)
    return cleaned


def _split_weak_single_source_thread(
    thread: StoryThread,
    config: AppConfig,
) -> list[StoryThread]:
    """Break apart single-source threads whose titles do not share a clear event anchor."""

    if len(thread.articles) > config.thread_clustering.max_articles_per_thread:
        return [thread]
    if thread.source_count != 1 or len(thread.articles) <= 2:
        return [thread]
    if _has_strong_shared_anchor(thread.articles):
        return [thread]
    LOGGER.info(
        "Splitting weak single-source thread '%s' into singleton threads",
        thread.topic,
    )
    return [
        _make_story_thread(
            topic=_fallback_topic(article.title),
            topic_en=article.title[:40].strip() or thread.topic_en,
            rationale="单源文章缺少明确共同事件锚点，已拆分",
            articles=[article],
            config=config,
        )
        for article in thread.articles
    ]


def _has_strong_shared_anchor(articles: Sequence[Article]) -> bool:
    """Return whether article titles share a meaningful event anchor."""

    normalized_titles = [_normalize_title(article.title) for article in articles]
    token_sets = [set(_meaningful_tokens(title)) for title in normalized_titles]
    if not token_sets:
        return False
    shared = set.intersection(*token_sets)
    if shared:
        return True
    token_counter: Counter[str] = Counter()
    for token_set in token_sets:
        token_counter.update(token_set)
    return any(count >= max(2, len(articles) - 1) for count in token_counter.values())


def _meaningful_tokens(title: str) -> list[str]:
    """Extract title tokens that are informative enough for anchor checks."""

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "after",
        "amid",
        "into",
        "will",
        "this",
        "that",
        "what",
        "about",
        "your",
        "they",
        "have",
        "has",
        "are",
        "why",
        "how",
        "when",
        "news",
        "brief",
        "watch",
        "story",
        "report",
    }
    return [
        token
        for token in title.split()
        if len(token) >= 4 and token not in stopwords
    ]


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


def _extract_response_text(response: Any) -> str:
    """Extract assistant text from an OpenAI-compatible response object."""

    choices = getattr(response, "choices", [])
    if not choices:
        raise ValueError("LLM response did not contain any choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content was empty")
    return content


def _chunked(items: Sequence[Article], size: int) -> list[list[Article]]:
    """Split articles into equally sized chunks."""

    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _create_client(config: AppConfig) -> Any:
    """Create an OpenAI-compatible client for DeepSeek."""

    from openai import OpenAI

    api_key = os.getenv(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{config.llm.api_key_env}' is required")
    return OpenAI(api_key=api_key, base_url=config.llm.base_url)
