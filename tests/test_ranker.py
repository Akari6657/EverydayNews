"""Tests for experimental story-thread ranking."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.models import StoryThread
from src.ranker import rank_threads


def test_rank_threads_prefers_multi_source_recent_threads(sample_config, make_article) -> None:
    """Ranking should place multi-source recent threads ahead of weaker threads."""

    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    major_article = make_article(
        guid="major-1",
        source_name="New York Times",
        source_slug="nyt",
        published=now - timedelta(minutes=10),
    )
    minor_article = make_article(
        guid="minor-1",
        title="Minor local update",
        source_name="BBC News",
        source_slug="bbc",
        published=now - timedelta(hours=6),
    )
    major_thread = StoryThread(
        thread_id=1,
        topic="重大事件",
        topic_en="Major story",
        articles=[major_article],
        source_names=["New York Times", "BBC News", "NPR"],
        source_count=3,
        primary=major_article,
        latest_published=major_article.published,
        rationale="测试",
    )
    minor_thread = StoryThread(
        thread_id=2,
        topic="普通新闻",
        topic_en="Minor story",
        articles=[minor_article],
        source_names=["BBC News"],
        source_count=1,
        primary=minor_article,
        latest_published=minor_article.published,
        rationale="测试",
    )

    ranked = rank_threads([minor_thread, major_thread], sample_config, now=now)

    assert ranked[0].topic == "重大事件"


def test_rank_threads_respects_floor_for_single_source_threads(sample_config, make_article) -> None:
    """Old low-signal single-source threads should fall below the ranking floor."""

    config = replace(
        sample_config,
        ranking=replace(sample_config.ranking, importance_floor=0.2, keep_major_always=True),
    )
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    stale_article = make_article(
        guid="stale-1",
        published=now - timedelta(hours=23),
    )
    stale_thread = StoryThread(
        thread_id=1,
        topic="陈旧新闻",
        topic_en="Stale story",
        articles=[stale_article],
        source_names=["New York Times"],
        source_count=1,
        primary=stale_article,
        latest_published=stale_article.published,
        rationale="测试",
    )

    ranked = rank_threads([stale_thread], config, now=now)

    assert ranked == []
