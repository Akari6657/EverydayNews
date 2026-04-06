"""Tests for fetch-stage keyword and category filtering."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from src import fetcher
from src.models import FeedConfig, SourceConfig


def test_fetch_all_feeds_drops_entries_matching_exclude_keywords(sample_config, monkeypatch) -> None:
    """Entries containing configured exclude keywords should be filtered out."""

    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    config = replace(
        sample_config,
        sources=[
            SourceConfig(
                name="Example News",
                slug="example",
                feeds=[
                    FeedConfig(
                        url="https://example.com/rss.xml",
                        category="top",
                        exclude_keywords=["gossip"],
                    )
                ],
            )
        ],
    )

    monkeypatch.setattr(fetcher, "_download_feed", lambda url: b"example")
    monkeypatch.setattr(
        fetcher,
        "_parse_feed_bytes",
        lambda raw_bytes: {
            "entries": [
                {
                    "title": "Celebrity gossip roundup",
                    "link": "https://example.com/gossip",
                    "summary": "A noisy entertainment story",
                    "published": "Sun, 06 Apr 2026 11:00:00 GMT",
                    "guid": "gossip-guid",
                },
                {
                    "title": "Markets steady after central bank comments",
                    "link": "https://example.com/markets",
                    "summary": "A meaningful economic story",
                    "published": "Sun, 06 Apr 2026 10:00:00 GMT",
                    "guid": "markets-guid",
                },
            ]
        },
    )

    articles = fetcher.fetch_all_feeds(config, now=now)

    assert [article.guid for article in articles] == ["markets-guid"]


def test_fetch_all_feeds_matches_keywords_against_normalized_links(sample_config, monkeypatch) -> None:
    """Keyword filters should also match normalized link slugs like heres-the-latest."""

    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    config = replace(
        sample_config,
        sources=[
            SourceConfig(
                name="Example News",
                slug="example",
                feeds=[
                    FeedConfig(
                        url="https://example.com/rss.xml",
                        category="top",
                        exclude_keywords=["here's the latest"],
                    )
                ],
            )
        ],
    )

    monkeypatch.setattr(fetcher, "_download_feed", lambda url: b"example")
    monkeypatch.setattr(
        fetcher,
        "_parse_feed_bytes",
        lambda raw_bytes: {
            "entries": [
                {
                    "title": "Iran war latest coverage",
                    "link": "https://example.com/live/iran-war/heres-the-latest",
                    "summary": "A live wrapper story",
                    "published": "Sun, 06 Apr 2026 11:00:00 GMT",
                    "guid": "latest-guid",
                },
                {
                    "title": "Cabinet reaches budget compromise",
                    "link": "https://example.com/politics/budget-deal",
                    "summary": "A substantive article",
                    "published": "Sun, 06 Apr 2026 10:00:00 GMT",
                    "guid": "budget-guid",
                },
            ]
        },
    )

    articles = fetcher.fetch_all_feeds(config, now=now)

    assert [article.guid for article in articles] == ["budget-guid"]


def test_fetch_all_feeds_drops_entries_matching_exclude_categories(sample_config, monkeypatch) -> None:
    """Entries tagged with excluded categories should be filtered out."""

    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    config = replace(
        sample_config,
        sources=[
            SourceConfig(
                name="Example News",
                slug="example",
                feeds=[
                    FeedConfig(
                        url="https://example.com/rss.xml",
                        category="top",
                        exclude_categories=["Page Six"],
                    )
                ],
            )
        ],
    )

    monkeypatch.setattr(fetcher, "_download_feed", lambda url: b"example")
    monkeypatch.setattr(
        fetcher,
        "_parse_feed_bytes",
        lambda raw_bytes: {
            "entries": [
                {
                    "title": "Tabloid exclusive",
                    "link": "https://example.com/tabloid",
                    "summary": "Should be filtered",
                    "published": "Sun, 06 Apr 2026 11:00:00 GMT",
                    "guid": "tabloid-guid",
                    "tags": [{"term": "Page Six"}],
                },
                {
                    "title": "Parliament approves budget",
                    "link": "https://example.com/budget",
                    "summary": "Should remain",
                    "published": "Sun, 06 Apr 2026 10:00:00 GMT",
                    "guid": "budget-guid",
                    "tags": [{"term": "Politics"}],
                },
            ]
        },
    )

    articles = fetcher.fetch_all_feeds(config, now=now)

    assert [article.guid for article in articles] == ["budget-guid"]
