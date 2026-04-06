"""Shared pytest fixtures for Daily Headline Agent tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.models import (
    AppConfig,
    Article,
    ArticleCluster,
    DedupConfig,
    EmailOutputConfig,
    FeedConfig,
    JsonOutputConfig,
    LLMConfig,
    MarkdownOutputConfig,
    OutputConfig,
    PipelineConfig,
    ScheduleConfig,
    SourceConfig,
    SummarizerConfig,
    SummarizerMapConfig,
    SummarizerReduceConfig,
    TelegramOutputConfig,
)


@pytest.fixture
def sample_config(tmp_path: Path) -> AppConfig:
    """Return a minimal app config for tests."""

    return AppConfig(
        sources=[
            SourceConfig(
                name="New York Times",
                slug="nyt",
                feeds=[FeedConfig(url="https://example.com/nyt.xml", category="top")],
            ),
            SourceConfig(
                name="BBC News",
                slug="bbc",
                feeds=[FeedConfig(url="https://example.com/bbc.xml", category="top")],
            ),
        ],
        pipeline=PipelineConfig(
            max_articles_per_source=10,
            total_articles_for_summary=5,
            importance_threshold=4,
            dedup_similarity_threshold=0.7,
            language="zh-CN",
            briefing_style="concise",
        ),
        dedup=DedupConfig(
            method="difflib",
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            similarity_threshold=0.7,
            clustering_algorithm="greedy",
            cache_embeddings=True,
        ),
        summarizer=SummarizerConfig(
            map=SummarizerMapConfig(batch_size=5, max_retries=2),
            reduce=SummarizerReduceConfig(top_k=30, max_retries=2),
        ),
        llm=LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            max_tokens=4096,
            temperature=0.3,
        ),
        output=OutputConfig(
            markdown=MarkdownOutputConfig(enabled=True, directory="output/md", group_by_month=True),
            json=JsonOutputConfig(enabled=True, directory="output/json", group_by_month=True),
            email=EmailOutputConfig(
                enabled=False,
                smtp_host="",
                smtp_port=587,
                sender="",
                recipients=[],
            ),
            telegram=TelegramOutputConfig(
                enabled=False,
                bot_token_env="TELEGRAM_BOT_TOKEN",
                chat_id_env="TELEGRAM_CHAT_ID",
            ),
        ),
        schedule=ScheduleConfig(timezone="Asia/Shanghai", run_at="08:00"),
        root_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
    )


@pytest.fixture
def make_article():
    """Create article test data with sensible defaults."""

    def _make_article(**overrides) -> Article:
        published = overrides.pop("published", datetime.now(timezone.utc) - timedelta(hours=1))
        values = {
            "title": "Markets rally as inflation cools",
            "description": "Stocks rose after the latest inflation report.",
            "link": "https://example.com/story",
            "source_name": "New York Times",
            "source_slug": "nyt",
            "category": "top",
            "published": published,
            "guid": "guid-1",
        }
        values.update(overrides)
        return Article(**values)

    return _make_article


@pytest.fixture
def make_cluster(make_article):
    """Create article clusters for map-stage summarizer tests."""

    def _make_cluster(cluster_id: str = "cluster-1", primary: Article | None = None, duplicates=None) -> ArticleCluster:
        primary_article = primary or make_article(guid=f"{cluster_id}-primary")
        duplicate_articles = list(duplicates or [])
        return ArticleCluster(
            cluster_id=cluster_id,
            primary=primary_article,
            duplicates=duplicate_articles,
        )

    return _make_cluster
