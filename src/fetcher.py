"""Fetch and normalize recent RSS articles from configured sources."""

from __future__ import annotations

import calendar
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any

from .models import AppConfig, Article, SourceConfig

LOGGER = logging.getLogger(__name__)
FETCH_TIMEOUT_SECONDS = 10
HTML_TAG_RE = re.compile(r"<[^>]+>")


def fetch_all_feeds(
    config: AppConfig,
    now: datetime | None = None,
) -> list[Article]:
    """Fetch articles from every configured source."""

    reference_time = now or datetime.now(timezone.utc)
    articles: list[Article] = []
    for source in config.sources:
        articles.extend(_fetch_source_articles(source, config, reference_time))
    return sorted(articles, key=lambda article: article.published, reverse=True)


def _fetch_source_articles(
    source: SourceConfig,
    config: AppConfig,
    reference_time: datetime,
) -> list[Article]:
    """Fetch and cap articles for a single source."""

    articles: list[Article] = []
    for feed in source.feeds:
        articles.extend(_fetch_feed_articles(source, feed.url, feed.category, reference_time))
    articles.sort(key=lambda article: article.published, reverse=True)
    return articles[: config.pipeline.max_articles_per_source]


def _fetch_feed_articles(
    source: SourceConfig,
    url: str,
    category: str,
    reference_time: datetime,
) -> list[Article]:
    """Fetch and parse one RSS feed."""

    try:
        raw_bytes = _download_feed(url)
        parsed_feed = _parse_feed_bytes(raw_bytes)
        return _entries_to_articles(parsed_feed, source, category, reference_time)
    except Exception as exc:
        LOGGER.warning("Failed to fetch feed %s: %s", url, exc)
        return []


def _download_feed(url: str) -> bytes:
    """Download feed content with a timeout."""

    import requests

    response = requests.get(url, timeout=FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _parse_feed_bytes(raw_bytes: bytes) -> Any:
    """Parse RSS XML bytes into a feedparser result."""

    import feedparser

    return feedparser.parse(raw_bytes)


def _entries_to_articles(
    parsed_feed: Any,
    source: SourceConfig,
    category: str,
    reference_time: datetime,
) -> list[Article]:
    """Convert feed entries into recent article objects."""

    articles: list[Article] = []
    cutoff = reference_time - timedelta(hours=24)
    for entry in _get_entries(parsed_feed):
        article = _entry_to_article(entry, source, category, reference_time)
        if article and article.published >= cutoff:
            articles.append(article)
    return articles


def _get_entries(parsed_feed: Any) -> list[Any]:
    """Return parsed feed entries from dict-like or object-like results."""

    if isinstance(parsed_feed, dict):
        return list(parsed_feed.get("entries", []))
    return list(getattr(parsed_feed, "entries", []))


def _entry_to_article(
    entry: Any,
    source: SourceConfig,
    category: str,
    reference_time: datetime,
) -> Article | None:
    """Convert a single feed entry into an article."""

    title = _get_field(entry, "title")
    link = _get_field(entry, "link")
    if not title or not link:
        return None
    description = _strip_html(_get_field(entry, "summary") or _get_field(entry, "description"))
    published = _parse_published(entry, reference_time)
    guid = _get_field(entry, "id") or _get_field(entry, "guid") or link
    return Article(
        title=title,
        description=description,
        link=link,
        source_name=source.name,
        source_slug=source.slug,
        category=category,
        published=published,
        guid=guid,
    )


def _get_field(entry: Any, key: str) -> str:
    """Read a field from dict-like or attribute-based objects."""

    if isinstance(entry, dict):
        value = entry.get(key, "")
    else:
        value = getattr(entry, key, "")
    return str(value).strip()


def _parse_published(entry: Any, fallback_time: datetime) -> datetime:
    """Parse entry publish time or fall back to the current time."""

    struct_time = _get_time_field(entry, "published_parsed") or _get_time_field(entry, "updated_parsed")
    if struct_time:
        timestamp = calendar.timegm(struct_time)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    for field in ("published", "updated"):
        value = _get_field(entry, field)
        if not value:
            continue
        try:
            return _to_utc(parsedate_to_datetime(value))
        except (TypeError, ValueError, IndexError):
            continue
    return _to_utc(fallback_time)


def _get_time_field(entry: Any, key: str) -> Any:
    """Read a time-struct field from an entry."""

    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)


def _to_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _strip_html(text: str) -> str:
    """Remove simple HTML markup from feed summaries."""

    without_tags = HTML_TAG_RE.sub(" ", text)
    collapsed = " ".join(unescape(without_tags).split())
    return collapsed.strip()
