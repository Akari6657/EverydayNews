"""Fetch and normalize recent RSS articles from configured sources."""

from __future__ import annotations

import calendar
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any

from .models import AppConfig, Article, FeedConfig, SourceConfig

LOGGER = logging.getLogger(__name__)
FETCH_TIMEOUT_SECONDS = 10
MAX_FETCH_WORKERS = 8
HTML_TAG_RE = re.compile(r"<[^>]+>")


def fetch_all_feeds(
    config: AppConfig,
    now: datetime | None = None,
) -> list[Article]:
    """Fetch articles from every configured source."""

    reference_time = now or datetime.now(timezone.utc)
    articles_by_source = _fetch_articles_by_source(config, reference_time)
    articles: list[Article] = []
    for source in config.sources:
        source_articles = _cap_source_articles(
            articles_by_source.get(source.slug, []),
            config.pipeline.max_articles_per_source,
        )
        articles.extend(source_articles)
    return sorted(articles, key=lambda article: article.published, reverse=True)


def _fetch_articles_by_source(
    config: AppConfig,
    reference_time: datetime,
) -> dict[str, list[Article]]:
    """Fetch all feeds concurrently and group results by source slug."""

    articles_by_source = {source.slug: [] for source in config.sources}
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_feed_articles, source, feed, reference_time): (source.slug, feed.url)
            for source in config.sources
            for feed in source.feeds
        }
        for future in as_completed(futures):
            source_slug, feed_url = futures[future]
            try:
                articles_by_source[source_slug].extend(future.result())
            except Exception as exc:
                LOGGER.warning("Failed to fetch feed %s: %s", feed_url, exc)
    return articles_by_source


def _cap_source_articles(
    articles: list[Article],
    max_articles_per_source: int,
) -> list[Article]:
    """Sort and cap articles for one source."""

    articles.sort(key=lambda article: article.published, reverse=True)
    return articles[:max_articles_per_source]


def _fetch_feed_articles(
    source: SourceConfig,
    feed: FeedConfig,
    reference_time: datetime,
) -> list[Article]:
    """Fetch and parse one RSS feed."""

    try:
        raw_bytes = _download_feed(feed.url)
        parsed_feed = _parse_feed_bytes(raw_bytes)
        return _entries_to_articles(parsed_feed, source, feed, reference_time)
    except Exception as exc:
        LOGGER.warning("Failed to fetch feed %s: %s", feed.url, exc)
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
    feed: FeedConfig,
    reference_time: datetime,
) -> list[Article]:
    """Convert feed entries into recent article objects."""

    articles: list[Article] = []
    cutoff = reference_time - timedelta(hours=24)
    for entry in _get_entries(parsed_feed):
        article = _entry_to_article(entry, source, feed, reference_time)
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
    feed: FeedConfig,
    reference_time: datetime,
) -> Article | None:
    """Convert a single feed entry into an article."""

    title = _get_field(entry, "title")
    link = _get_field(entry, "link")
    if not title or not link:
        return None
    description = _strip_html(_get_field(entry, "summary") or _get_field(entry, "description"))
    if _should_exclude_entry(entry, feed, title, description):
        return None
    published = _parse_published(entry, reference_time)
    guid = _get_field(entry, "id") or _get_field(entry, "guid") or link
    return Article(
        title=title,
        description=description,
        link=link,
        source_name=source.name,
        source_slug=source.slug,
        category=feed.category,
        published=published,
        guid=guid,
    )


def _should_exclude_entry(
    entry: Any,
    feed: FeedConfig,
    title: str,
    description: str,
) -> bool:
    """Return whether the entry should be filtered before article creation."""

    link = _get_field(entry, "link")
    haystack = _normalize_filter_text(f"{title}\n{description}\n{link}")
    if any(_normalize_filter_text(keyword) in haystack for keyword in feed.exclude_keywords):
        return True
    entry_categories = {value.casefold() for value in _entry_categories(entry)}
    for category in feed.exclude_categories:
        if category.casefold() in entry_categories:
            return True
    return False


def _entry_categories(entry: Any) -> list[str]:
    """Extract category-like labels from RSS entry metadata."""

    categories: list[str] = []
    primary_category = _get_field(entry, "category")
    if primary_category:
        categories.append(primary_category)
    tags = entry.get("tags", []) if isinstance(entry, dict) else getattr(entry, "tags", [])
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                value = tag.get("term") or tag.get("label") or tag.get("name") or ""
            else:
                value = (
                    getattr(tag, "term", "")
                    or getattr(tag, "label", "")
                    or getattr(tag, "name", "")
                )
            text = str(value).strip()
            if text:
                categories.append(text)
    return categories


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


def _normalize_filter_text(text: str) -> str:
    """Normalize text for resilient keyword matching."""

    sanitized = text.casefold().replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", sanitized).split())
