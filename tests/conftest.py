"""Shared pytest fixtures for Daily Headline Agent tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.models import (
    AppConfig,
    Article,
    DedupConfig,
    EmailOutputConfig,
    FeedConfig,
    JsonOutputConfig,
    LLMConfig,
    MarkdownOutputConfig,
    OutputConfig,
    PipelineConfig,
    RankingConfig,
    ScheduleConfig,
    SourceConfig,
    SummarizerConfig,
    SummarizerMapConfig,
    SummarizerReduceConfig,
    TelegramOutputConfig,
    ThreadClusteringConfig,
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
            importance_threshold=4,
            exclude_summary_keywords=["最新动态", "持续更新"],
        ),
        dedup=DedupConfig(
            method="difflib",
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_embeddings=True,
            within_thread_similarity_threshold=0.88,
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
            markdown=MarkdownOutputConfig(directory="output/md", group_by_month=True),
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
        thread_clustering=ThreadClusteringConfig(
            provider="deepseek",
            model="deepseek-chat",
            max_retries=2,
            max_articles_per_call=150,
            max_articles_per_thread=12,
            max_refinement_rounds=1,
        ),
        ranking=RankingConfig(
            importance_floor=0.0,
            keep_major_always=True,
        ),
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
