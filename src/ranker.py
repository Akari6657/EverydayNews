"""Rank experimental story threads before map-stage summarization."""

from __future__ import annotations

from datetime import datetime, timezone

from .models import AppConfig, StoryThread


def rank_threads(
    threads: list[StoryThread],
    config: AppConfig,
    now: datetime | None = None,
) -> list[StoryThread]:
    """Sort and filter threads with a lightweight composite priority score."""

    reference_time = now or datetime.now(timezone.utc)
    ranked = sorted(
        threads,
        key=lambda thread: (
            -thread_priority(thread, reference_time, config),
            -thread.source_count,
            -thread.latest_published.timestamp(),
            thread.topic,
        ),
    )
    kept: list[StoryThread] = []
    for thread in ranked:
        priority = thread_priority(thread, reference_time, config)
        if config.ranking.keep_major_always and thread.is_multi_source:
            kept.append(thread)
            continue
        if priority >= config.ranking.importance_floor:
            kept.append(thread)
    return kept


def thread_priority(thread: StoryThread, now: datetime, config: AppConfig) -> float:
    """Return a 0-1 priority score from source coverage and recency."""

    source_score = min((thread.source_count - 1) / 2.0, 1.0) * config.ranking.source_weight
    recency_score = _recency_factor(thread.latest_published, now) * config.ranking.recency_weight
    return source_score + recency_score


def _recency_factor(published: datetime, reference_time: datetime) -> float:
    """Return a 0-1 recency score within the 24-hour fetch window."""

    age_seconds = max(0.0, (reference_time - published).total_seconds())
    window_seconds = 24 * 60 * 60
    return max(0.0, 1.0 - min(age_seconds, window_seconds) / window_seconds)
