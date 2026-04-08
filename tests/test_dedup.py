"""Tests for title-based deduplication and GUID cache handling."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.dedup import deduplicate_articles, deduplicate_with_diagnostics


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


def test_deduplicate_with_diagnostics_reports_seen_and_limit(sample_config, make_article, tmp_path) -> None:
    """Dedup diagnostics should surface seen-cache filtering and cluster truncation."""

    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    config = replace(
        sample_config,
        pipeline=replace(sample_config.pipeline, total_articles_for_summary=1),
    )
    cache_path = tmp_path / "seen.json"
    cache_path.write_text(json.dumps({"seen-guid": now.isoformat()}), encoding="utf-8")
    clusters, diagnostics = deduplicate_with_diagnostics(
        [
            make_article(guid="seen-guid", title="Already sent"),
            make_article(guid="fresh-guid-1", title="Fresh one"),
            make_article(guid="fresh-guid-2", title="Another different story"),
        ],
        config,
        cache_path=cache_path,
        now=now,
    )

    assert len(clusters) == 1
    assert diagnostics.seen_filtered == 1
    assert diagnostics.fresh_articles == 2
    assert diagnostics.clusters_before_limit == 2
    assert diagnostics.clusters_after_limit == 1
