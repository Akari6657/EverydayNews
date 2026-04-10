"""Tests for strict within-thread near-duplicate cleanup."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.dedup import deduplicate_within_thread_with_diagnostics
from src.models import StoryThread


class FakeEncoder:
    """Return deterministic embeddings for within-thread dedup tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Map each embedding text to a pre-baked vector."""

        return [self.vectors[text] for text in texts]


def test_within_thread_dedup_merges_near_duplicate_articles(sample_config, make_article) -> None:
    """Near-identical reports inside one thread should collapse to one canonical article."""

    config = replace(
        sample_config,
        dedup=replace(
            sample_config.dedup,
            method="embedding",
            within_thread_enabled=True,
            within_thread_similarity_threshold=0.88,
        ),
    )
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    article_a = make_article(
        guid="a1",
        title="Oil prices plunge after ceasefire deal",
        description="Markets rally as the truce calms traders.",
        source_name="New York Times",
        source_slug="nyt",
        published=now,
    )
    article_b = make_article(
        guid="a2",
        title="Oil prices plunge after ceasefire deal",
        description="Markets rally as the truce calms traders.",
        source_name="BBC News",
        source_slug="bbc",
        published=now - timedelta(minutes=1),
    )
    article_c = make_article(
        guid="a3",
        title="Fuel costs may stay elevated for weeks",
        description="Consumers may not see relief immediately.",
        source_name="Reuters",
        source_slug="reuters",
        published=now - timedelta(minutes=2),
    )
    thread = StoryThread(
        thread_id=1,
        topic="市场影响",
        topic_en="Market impact",
        articles=[article_a, article_b, article_c],
        source_names=["New York Times", "BBC News", "Reuters"],
        source_count=3,
        primary=article_a,
        latest_published=now,
        rationale="测试",
    )
    encoder = FakeEncoder(
        {
            "Oil prices plunge after ceasefire deal\nMarkets rally as the truce calms traders.": [1.0, 0.0],
            "Fuel costs may stay elevated for weeks\nConsumers may not see relief immediately.": [0.0, 1.0],
        }
    )

    deduplicated, diagnostics = deduplicate_within_thread_with_diagnostics(
        thread,
        config,
        encoder=encoder,
        now=now,
    )

    assert len(deduplicated.articles) == 2
    assert deduplicated.primary.guid == "a1"
    assert deduplicated.source_count == 3
    assert diagnostics.before_articles == 3
    assert diagnostics.after_articles == 2
    assert len(diagnostics.merged_pairs) == 1
    assert diagnostics.merged_pairs[0].removed_article.guid == "a2"


def test_within_thread_dedup_keeps_distinct_angles(sample_config, make_article) -> None:
    """Distinct angles inside one thread should not be merged away."""

    config = replace(
        sample_config,
        dedup=replace(
            sample_config.dedup,
            method="embedding",
            within_thread_enabled=True,
            within_thread_similarity_threshold=0.88,
        ),
    )
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    article_a = make_article(
        guid="a1",
        title="Ceasefire terms leave uranium questions unresolved",
        description="Negotiators still dispute enrichment monitoring.",
        published=now,
    )
    article_b = make_article(
        guid="a2",
        title="Pakistan emerges as quiet broker in ceasefire talks",
        description="Diplomatic channels remained active overnight.",
        source_name="BBC News",
        source_slug="bbc",
        published=now - timedelta(minutes=1),
    )
    thread = StoryThread(
        thread_id=1,
        topic="停火条款与谈判",
        topic_en="Ceasefire terms and negotiations",
        articles=[article_a, article_b],
        source_names=["New York Times", "BBC News"],
        source_count=2,
        primary=article_a,
        latest_published=now,
        rationale="测试",
    )
    encoder = FakeEncoder(
        {
            "Ceasefire terms leave uranium questions unresolved\nNegotiators still dispute enrichment monitoring.": [1.0, 0.0],
            "Pakistan emerges as quiet broker in ceasefire talks\nDiplomatic channels remained active overnight.": [0.3, 0.7],
        }
    )

    deduplicated, diagnostics = deduplicate_within_thread_with_diagnostics(
        thread,
        config,
        encoder=encoder,
        now=now,
    )

    assert len(deduplicated.articles) == 2
    assert diagnostics.before_articles == 2
    assert diagnostics.after_articles == 2
    assert diagnostics.merged_pairs == []
