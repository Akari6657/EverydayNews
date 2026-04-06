"""Deduplicate articles by title similarity and cross-run GUID cache."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

from .models import AppConfig, Article

LOGGER = logging.getLogger(__name__)
NON_WORD_RE = re.compile(r"[^\w\s]+")


def deduplicate_articles(
    articles: list[Article],
    config: AppConfig,
    cache_path: str | Path | None = None,
    now: datetime | None = None,
) -> list[Article]:
    """Remove duplicate and previously seen articles."""

    reference_time = now or datetime.now(timezone.utc)
    cache_file = Path(cache_path) if cache_path else config.root_dir / "cache" / "seen_guids.json"
    priorities = config.source_priorities()
    seen_cache = _prune_cache(_load_cache(cache_file), reference_time)
    selected = _pick_unique_articles(articles, priorities, config.pipeline.dedup_similarity_threshold, seen_cache)
    selected.sort(key=lambda article: article.published, reverse=True)
    final_articles = selected[: config.pipeline.total_articles_for_summary]
    updated_cache = _update_cache(seen_cache, final_articles, reference_time)
    _save_cache(cache_file, updated_cache)
    return final_articles


def _pick_unique_articles(
    articles: list[Article],
    priorities: dict[str, int],
    threshold: float,
    seen_cache: dict[str, str],
) -> list[Article]:
    """Keep only unseen, non-duplicate articles."""

    kept: list[Article] = []
    ordered = sorted(articles, key=lambda article: (_priority(article, priorities), -article.published.timestamp()))
    for article in ordered:
        if article.guid in seen_cache:
            continue
        if _matches_existing(article, kept, threshold):
            continue
        kept.append(article)
    return kept


def _priority(article: Article, priorities: dict[str, int]) -> int:
    """Return source priority with a safe default."""

    return priorities.get(article.source_slug, len(priorities))


def _matches_existing(article: Article, kept: list[Article], threshold: float) -> bool:
    """Check whether an article is too similar to a kept one."""

    normalized = _normalize_title(article.title)
    return any(_similarity(normalized, _normalize_title(item.title)) > threshold for item in kept)


def _normalize_title(title: str) -> str:
    """Normalize punctuation and spacing for fuzzy comparison."""

    lowered = title.casefold()
    stripped = NON_WORD_RE.sub(" ", lowered)
    return " ".join(stripped.split())


def _similarity(left: str, right: str) -> float:
    """Return a SequenceMatcher ratio between two titles."""

    return SequenceMatcher(a=left, b=right).ratio()


def _load_cache(cache_path: Path) -> dict[str, str]:
    """Load GUID cache from disk."""

    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Cache file is invalid JSON, rebuilding: %s", cache_path)
        return {}
    if not isinstance(payload, dict):
        LOGGER.warning("Cache file is not a mapping, rebuilding: %s", cache_path)
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _prune_cache(cache: dict[str, str], reference_time: datetime) -> dict[str, str]:
    """Keep only GUIDs from the last seven days."""

    cutoff = reference_time - timedelta(days=7)
    pruned: dict[str, str] = {}
    for guid, timestamp in cache.items():
        parsed = _parse_cache_time(timestamp)
        if parsed and parsed >= cutoff:
            pruned[guid] = parsed.isoformat()
    return pruned


def _parse_cache_time(timestamp: str) -> datetime | None:
    """Parse cached ISO timestamps safely."""

    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _update_cache(
    cache: dict[str, str],
    articles: list[Article],
    reference_time: datetime,
) -> dict[str, str]:
    """Add newly used GUIDs to the existing cache."""

    updated = dict(cache)
    for article in articles:
        updated[article.guid] = reference_time.isoformat()
    return _prune_cache(updated, reference_time)


def _save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    """Persist GUID cache to disk."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
