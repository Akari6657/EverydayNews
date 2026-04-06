"""Tests for cluster ordering with cross-source priority signals."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.dedup import deduplicate
from src.models import FeedConfig, SourceConfig


class FakeEncoder:
    """Deterministic encoder for cluster ordering tests."""

    def __init__(self, vectors_by_title: dict[str, list[float]]):
        self.vectors_by_title = vectors_by_title

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            title = text.split("\n", 1)[0]
            vectors.append(self.vectors_by_title[title])
        return vectors


def test_cross_source_cluster_priority_beats_single_source_recency(sample_config, make_article, tmp_path) -> None:
    """A widely reported cluster should outrank a newer single-source cluster."""

    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    config = replace(
        sample_config,
        sources=[
            SourceConfig(name="Source 1", slug="s1", feeds=[FeedConfig(url="https://example.com/1", category="top")]),
            SourceConfig(name="Source 2", slug="s2", feeds=[FeedConfig(url="https://example.com/2", category="top")]),
            SourceConfig(name="Source 3", slug="s3", feeds=[FeedConfig(url="https://example.com/3", category="top")]),
            SourceConfig(name="Source 4", slug="s4", feeds=[FeedConfig(url="https://example.com/4", category="top")]),
            SourceConfig(name="Source 5", slug="s5", feeds=[FeedConfig(url="https://example.com/5", category="top")]),
            SourceConfig(name="Source 6", slug="s6", feeds=[FeedConfig(url="https://example.com/6", category="top")]),
        ],
        pipeline=replace(sample_config.pipeline, total_articles_for_summary=10),
        dedup=replace(sample_config.dedup, method="embedding", similarity_threshold=0.8),
    )
    encoder = FakeEncoder(
        {
            "Big event source 1": [1.0, 0.0],
            "Big event source 2": [0.99, 0.01],
            "Big event source 3": [0.98, 0.02],
            "Big event source 4": [0.97, 0.03],
            "Big event source 5": [0.96, 0.04],
            "Fresh minor event": [0.0, 1.0],
        }
    )
    articles = [
        make_article(
            title="Big event source 1",
            guid="s1-guid",
            source_name="Source 1",
            source_slug="s1",
            published=now - timedelta(hours=20),
        ),
        make_article(
            title="Big event source 2",
            guid="s2-guid",
            source_name="Source 2",
            source_slug="s2",
            published=now - timedelta(hours=20),
        ),
        make_article(
            title="Big event source 3",
            guid="s3-guid",
            source_name="Source 3",
            source_slug="s3",
            published=now - timedelta(hours=19),
        ),
        make_article(
            title="Big event source 4",
            guid="s4-guid",
            source_name="Source 4",
            source_slug="s4",
            published=now - timedelta(hours=19),
        ),
        make_article(
            title="Big event source 5",
            guid="s5-guid",
            source_name="Source 5",
            source_slug="s5",
            published=now - timedelta(hours=18),
        ),
        make_article(
            title="Fresh minor event",
            guid="s6-guid",
            source_name="Source 6",
            source_slug="s6",
            published=now - timedelta(minutes=10),
        ),
    ]

    clusters = deduplicate(
        articles,
        config,
        cache_path=tmp_path / "seen.json",
        embedding_cache_path=tmp_path / "embeddings.pkl",
        now=now,
        encoder=encoder,
    )

    assert len(clusters) == 2
    assert clusters[0].source_count == 5
    assert clusters[1].primary.title == "Fresh minor event"
