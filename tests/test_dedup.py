"""Tests for title-based deduplication and GUID cache handling."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.dedup import deduplicate_articles


def test_deduplicate_articles_prefers_higher_priority_source(sample_config, make_article, tmp_path) -> None:
    """Dedup should keep the source that appears earlier in config."""

    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    nyt_article = make_article(
        title="Markets rally after inflation cools",
        guid="nyt-guid",
        source_name="New York Times",
        source_slug="nyt",
        published=now - timedelta(hours=2),
    )
    bbc_article = make_article(
        title="Markets rally after inflation cools in new report",
        guid="bbc-guid",
        source_name="BBC News",
        source_slug="bbc",
        published=now - timedelta(hours=1),
    )

    results = deduplicate_articles(
        [bbc_article, nyt_article],
        sample_config,
        cache_path=tmp_path / "seen.json",
        now=now,
    )

    assert [article.guid for article in results] == ["nyt-guid"]
    cache_payload = json.loads((tmp_path / "seen.json").read_text(encoding="utf-8"))
    assert "nyt-guid" in cache_payload


def test_deduplicate_articles_recovers_from_invalid_cache(sample_config, make_article, tmp_path) -> None:
    """Invalid cache JSON should not crash the deduplicator."""

    cache_path = tmp_path / "seen.json"
    cache_path.write_text("{invalid-json", encoding="utf-8")

    results = deduplicate_articles(
        [make_article(guid="guid-123")],
        sample_config,
        cache_path=cache_path,
        now=datetime.now(timezone.utc),
    )

    assert len(results) == 1
