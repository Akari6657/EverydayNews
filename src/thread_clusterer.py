"""LLM-based story-thread clustering for fetched RSS articles."""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from itertools import chain
from typing import Any, Sequence

from .llm_utils import chunked, create_client, extract_response_text, load_json_payload
from .models import AppConfig, Article, StoryThread
from .prompts import (
    THREAD_CLUSTERING_JSON_RETRY_SUFFIX,
    THREAD_CLUSTERING_PROMPT_TEMPLATE,
    THREAD_CLUSTERING_SYSTEM_PROMPT,
    THREAD_MERGE_JSON_RETRY_SUFFIX,
    THREAD_MERGE_PROMPT_TEMPLATE,
    THREAD_REFINEMENT_PROMPT_TEMPLATE,
)

LOGGER = logging.getLogger(__name__)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def cluster_into_threads(
    articles: list[Article],
    config: AppConfig,
    client: Any | None = None,
) -> list[StoryThread]:
    """Group fetched articles into story threads via one LLM clustering pass."""

    if not articles:
        return []
    wrapper_threads, clusterable_articles = _extract_wrapper_threads(articles, config)
    if not clusterable_articles:
        return _renumber_threads(_sort_threads(wrapper_threads))
    if len(clusterable_articles) > config.thread_clustering.max_articles_per_call:
        threads = _cluster_large_article_set(clusterable_articles, config, client=client)
        refined = _refine_threads(threads, config, client or create_client(config))
        return _renumber_threads(_sort_threads(refined + wrapper_threads))
    llm_client = client or create_client(config)
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
    threads = _merge_overlapping_threads(threads, config)
    refined_threads = _refine_threads(threads, config, llm_client)
    return _renumber_threads(_sort_threads(refined_threads + wrapper_threads))


def _cluster_large_article_set(
    articles: list[Article],
    config: AppConfig,
    client: Any | None,
) -> list[StoryThread]:
    """Cluster large article sets by chunking and renumbering the results."""

    max_per_call = config.thread_clustering.max_articles_per_call
    LOGGER.warning(
        "Thread clustering received %s articles; clustering in %s-sized chunks without merge pass",
        len(articles),
        max_per_call,
    )
    threads: list[StoryThread] = []
    for chunk in chunked(articles, max_per_call):
        threads.extend(cluster_into_threads(chunk, config, client=client))
    # Renumber before merge pass so every thread_id is unique across chunks
    threads = _renumber_threads(_sort_threads(threads))
    if config.thread_clustering.enable_chunk_merge:
        llm_client = client or create_client(config)
        threads = _merge_chunk_threads_via_llm(threads, config, llm_client)
    return threads


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

        raw_text = extract_response_text(response)
        try:
            payload = load_json_payload(raw_text)
            return _extract_threads(payload)
        except (ValueError, TypeError) as exc:
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
        normalized.startswith("watch")
        or normalized.startswith("morning news brief")
        or normalized.startswith("what to know")
        or normalized.startswith("what we know")
        or normalized.startswith("up first")
        or normalized.startswith("daily briefing")
        or normalized.startswith("evening briefing")
        or normalized.startswith("today in news")
        or normalized.startswith("week in review")
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


def _combine_thread_group(group: list[StoryThread], config: AppConfig) -> StoryThread:
    """Merge a list of StoryThreads into one, preferring the topic of the most-sourced thread."""

    primary = max(group, key=lambda t: (t.source_count, len(t.articles)))
    seen_urls: set[str] = set()
    combined_articles: list[Article] = []
    for thread in group:
        for article in thread.articles:
            if article.link not in seen_urls:
                seen_urls.add(article.link)
                combined_articles.append(article)
    return _make_story_thread(
        topic=primary.topic,
        topic_en=primary.topic_en,
        rationale=f"聚类后合并（{len(group)} 条线）",
        articles=combined_articles,
        config=config,
    )


def _merge_overlapping_threads(
    threads: list[StoryThread],
    config: AppConfig,
) -> list[StoryThread]:
    """Merge threads whose article titles share substantial token overlap (Jaccard ≥ threshold).

    Runs on the non-chunked path as a safety net after _post_process_threads.
    Uses union-find so transitive overlaps are handled correctly.
    """

    if len(threads) <= 1 or not config.thread_clustering.enable_post_merge:
        return threads

    threshold = config.thread_clustering.merge_overlap_threshold

    def _thread_tokens(t: StoryThread) -> frozenset[str]:
        return frozenset(chain.from_iterable(
            _meaningful_tokens(_normalize_title(article.title)) for article in t.articles
        ))

    token_sets = [_thread_tokens(t) for t in threads]

    parent = list(range(len(threads)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(threads)):
        for j in range(i + 1, len(threads)):
            a, b = token_sets[i], token_sets[j]
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(len(threads)):
        groups.setdefault(find(i), []).append(i)

    result: list[StoryThread] = []
    for indices in groups.values():
        group = [threads[i] for i in indices]
        if len(group) == 1:
            result.append(group[0])
        else:
            LOGGER.info(
                "Heuristic merge: combining %s overlapping threads: %s",
                len(group),
                [t.topic for t in group],
            )
            result.append(_combine_thread_group(group, config))
    return result


def _build_thread_topics_payload(threads: list[StoryThread]) -> str:
    """Render a compact thread-topic list for the LLM merge prompt."""

    return "\n".join(
        f"[{t.thread_id}] \"{t.topic}\" / {t.topic_en} "
        f"({len(t.articles)}篇, {' / '.join(t.source_names[:3])})"
        for t in threads
    )


def _merge_chunk_threads_via_llm(
    threads: list[StoryThread],
    config: AppConfig,
    client: Any,
) -> list[StoryThread]:
    """Ask the LLM to merge duplicate threads produced by independent chunk clustering.

    This is a best-effort pass — on any failure the original threads are returned unchanged.
    """

    if len(threads) <= 1 or not config.thread_clustering.enable_chunk_merge:
        return threads

    payload = _build_thread_topics_payload(threads)
    prompt = THREAD_MERGE_PROMPT_TEMPLATE.format(threads_payload=payload)
    for attempt in range(config.thread_clustering.max_retries):
        try:
            response = _request_threads(client, config, prompt)
            raw_text = extract_response_text(response)
            merge_data = load_json_payload(raw_text)
            break
        except Exception as exc:
            if attempt == config.thread_clustering.max_retries - 1:
                LOGGER.warning("Thread merge LLM call failed, keeping chunks separate: %s", exc)
                return threads
            LOGGER.warning("Thread merge attempt %s failed: %s", attempt + 1, exc)
            prompt = prompt + THREAD_MERGE_JSON_RETRY_SUFFIX
            time.sleep(2**attempt)

    merges = merge_data.get("merges", [])
    if not isinstance(merges, list) or not merges:
        return threads

    threads_by_id = {t.thread_id: t for t in threads}
    merged_ids: set[int] = set()
    result: list[StoryThread] = []

    for merge_spec in merges:
        ids = merge_spec.get("ids", [])
        if not isinstance(ids, list) or len(ids) < 2:
            continue
        group = [threads_by_id[i] for i in ids if i in threads_by_id]
        if len(group) < 2:
            continue
        combined = _combine_thread_group(group, config)
        new_topic = merge_spec.get("topic", "").strip()
        new_topic_en = merge_spec.get("topic_en", "").strip()
        if new_topic:
            combined = StoryThread(
                thread_id=combined.thread_id,
                topic=new_topic,
                topic_en=new_topic_en or combined.topic_en,
                articles=combined.articles,
                source_names=combined.source_names,
                source_count=combined.source_count,
                primary=combined.primary,
                latest_published=combined.latest_published,
                rationale=combined.rationale,
            )
        result.append(combined)
        merged_ids.update(ids)
        LOGGER.info("LLM merge pass: combined threads %s → '%s'", ids, combined.topic)

    for t in threads:
        if t.thread_id not in merged_ids:
            result.append(t)

    return result
