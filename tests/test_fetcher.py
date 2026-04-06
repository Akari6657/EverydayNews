"""Tests for RSS fetching and normalization."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from src import fetcher
from src.models import Article, FeedConfig, SourceConfig


def test_fetch_all_feeds_returns_recent_articles(sample_config, monkeypatch) -> None:
    """Fetcher should keep recent entries and strip HTML."""

    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)

    def fake_download(url: str) -> bytes:
        return url.encode("utf-8")

    def fake_parse(raw_bytes: bytes):
        if b"nyt" in raw_bytes:
            return {
                "entries": [
                    {
                        "title": "Fresh headline",
                        "link": "https://example.com/fresh",
                        "summary": "<p>Hello <b>world</b></p>",
                        "published": "Sun, 06 Apr 2026 11:00:00 GMT",
                        "guid": "fresh-guid",
                    },
                    {
                        "title": "Old headline",
                        "link": "https://example.com/old",
                        "summary": "<p>Old</p>",
                        "published": "Sat, 04 Apr 2026 11:00:00 GMT",
                        "guid": "old-guid",
                    },
                ]
            }
        return {"entries": []}

    monkeypatch.setattr(fetcher, "_download_feed", fake_download)
    monkeypatch.setattr(fetcher, "_parse_feed_bytes", fake_parse)

    articles = fetcher.fetch_all_feeds(sample_config, now=now)

    assert len(articles) == 1
    assert articles[0].title == "Fresh headline"
    assert articles[0].description == "Hello world"


def test_fetch_all_feeds_handles_feed_errors(sample_config, monkeypatch) -> None:
    """Fetcher should continue when a feed request fails."""

    def fake_download(url: str) -> bytes:
        raise RuntimeError("network down")

    monkeypatch.setattr(fetcher, "_download_feed", fake_download)

    articles = fetcher.fetch_all_feeds(sample_config, now=datetime.now(timezone.utc))

    assert articles == []


def test_fetch_all_feeds_merges_feed_results_before_source_cap(sample_config, monkeypatch) -> None:
    """Fetcher should merge all feeds for a source before applying the source cap."""

    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    config = replace(
        sample_config,
        sources=[
            SourceConfig(
                name="New York Times",
                slug="nyt",
                feeds=[
                    FeedConfig(url="https://example.com/nyt-home.xml", category="home"),
                    FeedConfig(url="https://example.com/nyt-world.xml", category="world"),
                ],
            )
        ],
        pipeline=replace(sample_config.pipeline, max_articles_per_source=2),
    )

    def make_article(guid: str, title: str, category: str, published_hour: int) -> Article:
        return Article(
            title=title,
            description=f"{title} summary",
            link=f"https://example.com/{guid}",
            source_name="New York Times",
            source_slug="nyt",
            category=category,
            published=datetime(2026, 4, 6, published_hour, 0, tzinfo=timezone.utc),
            guid=guid,
        )

    def fake_fetch(source, feed, reference_time):
        if feed.category == "home":
            return [
                make_article("home-new", "Home newest", "home", 11),
                make_article("home-old", "Home older", "home", 8),
            ]
        return [make_article("world-mid", "World middle", "world", 10)]

    monkeypatch.setattr(fetcher, "_fetch_feed_articles", fake_fetch)

    articles = fetcher.fetch_all_feeds(config, now=now)

    assert [article.guid for article in articles] == ["home-new", "world-mid"]
