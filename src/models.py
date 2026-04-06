"""Shared dataclasses for configuration and pipeline data."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class FeedConfig:
    """Configuration for a single RSS feed."""

    url: str
    category: str


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for a news source and its feeds."""

    name: str
    slug: str
    feeds: list[FeedConfig]


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline-level tuning options."""

    max_articles_per_source: int
    total_articles_for_summary: int
    dedup_similarity_threshold: float
    language: str
    briefing_style: str


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings."""

    provider: str
    model: str
    base_url: str
    api_key_env: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class MarkdownOutputConfig:
    """Markdown delivery settings."""

    enabled: bool
    directory: str


@dataclass(frozen=True)
class EmailOutputConfig:
    """Email delivery settings."""

    enabled: bool
    smtp_host: str
    smtp_port: int
    sender: str
    recipients: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TelegramOutputConfig:
    """Telegram delivery settings."""

    enabled: bool
    bot_token_env: str
    chat_id_env: str


@dataclass(frozen=True)
class OutputConfig:
    """All output channel settings."""

    markdown: MarkdownOutputConfig
    email: EmailOutputConfig
    telegram: TelegramOutputConfig


@dataclass(frozen=True)
class ScheduleConfig:
    """Scheduling configuration."""

    timezone: str
    run_at: str


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration."""

    sources: list[SourceConfig]
    pipeline: PipelineConfig
    llm: LLMConfig
    output: OutputConfig
    schedule: ScheduleConfig
    root_dir: Path
    config_path: Path

    def source_priorities(self) -> dict[str, int]:
        """Return source priority by config order."""

        return {source.slug: index for index, source in enumerate(self.sources)}


@dataclass(frozen=True)
class Article:
    """Normalized article data parsed from RSS feeds."""

    title: str
    description: str
    link: str
    source_name: str
    source_slug: str
    category: str
    published: datetime
    guid: str


@dataclass(frozen=True)
class BriefingResult:
    """Final model-generated briefing payload."""

    content: str
    model: str
    token_usage: dict[str, int]
    generated_at: datetime
