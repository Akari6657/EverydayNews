"""Tests for embedding-based semantic deduplication."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.dedup import deduplicate


class FakeEncoder:
    """Deterministic encoder for semantic dedup tests."""

    def __init__(self, vectors_by_title: dict[str, list[float]]):
        self.vectors_by_title = vectors_by_title
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            title = text.split("\n", 1)[0]
            vectors.append(self.vectors_by_title[title])
        return vectors


def test_deduplicate_clusters_semantically_similar_articles(sample_config, make_article, tmp_path) -> None:
    """Different phrasings of the same event should land in one cluster."""

    config = replace(
        sample_config,
        dedup=replace(sample_config.dedup, method="embedding", similarity_threshold=0.75),
    )
    encoder = FakeEncoder(
        {
            "Trump wins Pennsylvania": [1.0, 0.0, 0.0],
            "Pennsylvania goes red as Trump secures victory": [0.98, 0.02, 0.0],
            "Trump clinches key Pennsylvania battleground": [0.97, 0.03, 0.0],
        }
    )
    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    articles = [
        make_article(
            title="Trump wins Pennsylvania",
            guid="nyt-guid",
            source_name="New York Times",
            source_slug="nyt",
            published=now - timedelta(hours=3),
        ),
        make_article(
            title="Pennsylvania goes red as Trump secures victory",
            guid="bbc-guid",
            source_name="BBC News",
            source_slug="bbc",
            published=now - timedelta(hours=2),
        ),
        make_article(
            title="Trump clinches key Pennsylvania battleground",
            guid="guardian-guid",
            source_name="The Guardian",
            source_slug="guardian",
            published=now - timedelta(hours=1),
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

    assert len(clusters) == 1
    assert clusters[0].primary.guid == "nyt-guid"
    assert clusters[0].source_count == 3


def test_deduplicate_uses_embedding_cache_on_second_run(sample_config, make_article, tmp_path) -> None:
    """Embedding cache should avoid re-encoding the same GUID twice."""

    config = replace(
        sample_config,
        dedup=replace(sample_config.dedup, method="embedding", similarity_threshold=0.75),
    )
    encoder = FakeEncoder({"Markets rally as inflation cools": [1.0, 0.0, 0.0]})
    article = make_article(guid="guid-123")
    embedding_cache_path = tmp_path / "embeddings.pkl"

    deduplicate(
        [article],
        config,
        cache_path=tmp_path / "seen-first.json",
        embedding_cache_path=embedding_cache_path,
        now=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
        encoder=encoder,
    )
    deduplicate(
        [article],
        config,
        cache_path=tmp_path / "seen-second.json",
        embedding_cache_path=embedding_cache_path,
        now=datetime(2026, 4, 6, 13, 0, tzinfo=timezone.utc),
        encoder=encoder,
    )

    assert len(encoder.calls) == 1
    assert embedding_cache_path.exists()


def test_deduplicate_keeps_higher_priority_source_as_primary(sample_config, make_article, tmp_path) -> None:
    """When articles cluster together, the earlier config source should win."""

    config = replace(
        sample_config,
        dedup=replace(sample_config.dedup, method="embedding", similarity_threshold=0.75),
    )
    encoder = FakeEncoder(
        {
            "Markets rally after inflation cools": [1.0, 0.0, 0.0],
            "Inflation cools as markets surge": [0.99, 0.01, 0.0],
        }
    )
    now = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)
    articles = [
        make_article(
            title="Inflation cools as markets surge",
            guid="bbc-guid",
            source_name="BBC News",
            source_slug="bbc",
            published=now - timedelta(hours=1),
        ),
        make_article(
            title="Markets rally after inflation cools",
            guid="nyt-guid",
            source_name="New York Times",
            source_slug="nyt",
            published=now - timedelta(hours=3),
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

    assert len(clusters) == 1
    assert clusters[0].primary.source_slug == "nyt"
