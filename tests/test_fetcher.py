"""Tests for RSS fetching and normalization."""

from __future__ import annotations

from datetime import datetime, timezone

from src import fetcher


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
