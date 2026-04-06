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
    exclude_keywords: list[str] = field(default_factory=list)
    exclude_categories: list[str] = field(default_factory=list)


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
    importance_threshold: int
    max_items_per_topic: int
    exclude_summary_keywords: list[str]
    dedup_similarity_threshold: float
    language: str
    briefing_style: str


@dataclass(frozen=True)
class DedupConfig:
    """Deduplication strategy settings."""

    method: str
    model: str
    similarity_threshold: float
    clustering_algorithm: str
    cache_embeddings: bool


@dataclass(frozen=True)
class SummarizerMapConfig:
    """Map-stage summarization settings."""

    batch_size: int
    max_retries: int


@dataclass(frozen=True)
class SummarizerReduceConfig:
    """Reduce-stage summarization settings."""

    top_k: int
    max_retries: int


@dataclass(frozen=True)
class SummarizerConfig:
    """Map-reduce summarization configuration."""

    map: SummarizerMapConfig
    reduce: SummarizerReduceConfig


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
    group_by_month: bool


@dataclass(frozen=True)
class JsonOutputConfig:
    """Structured JSON output settings."""

    enabled: bool
    directory: str
    group_by_month: bool


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
    json: JsonOutputConfig
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
    dedup: DedupConfig
    summarizer: SummarizerConfig
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


@dataclass
class ArticleCluster:
    """A group of articles reporting the same event across sources."""

    cluster_id: str
    primary: Article
    duplicates: list[Article] = field(default_factory=list)
    source_count: int = 0
    source_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize metadata derived from cluster membership."""

        self.refresh_metadata()

    def add_duplicate(self, article: Article) -> None:
        """Append a duplicate article and refresh metadata."""

        self.duplicates.append(article)
        self.refresh_metadata()

    def refresh_metadata(self) -> None:
        """Refresh source metadata from all current articles."""

        distinct_names: list[str] = []
        for article in self.all_articles:
            if article.source_name not in distinct_names:
                distinct_names.append(article.source_name)
        self.source_names = distinct_names
        self.source_count = len(distinct_names)

    @property
    def all_articles(self) -> list[Article]:
        """Return the primary article followed by duplicates."""

        return [self.primary] + list(self.duplicates)


@dataclass(frozen=True)
class ClusterSummary:
    """Map-stage summary for a single deduplicated event cluster."""

    cluster_id: str
    topic: str
    headline_zh: str
    summary_zh: str
    importance: int
    entities: list[str]
    source_names: list[str]
    primary_link: str


@dataclass(frozen=True)
class MapSummariesResult:
    """Aggregated result of the map-stage summarization step."""

    summaries: list[ClusterSummary]
    token_usage: dict[str, int]
    model: str


@dataclass(frozen=True)
class FinalBriefing:
    """Reduce-stage structured daily briefing."""

    date: str
    overview_zh: str
    topics: dict[str, list[ClusterSummary]]
    total_clusters: int
    total_sources: int
    generated_at: datetime
    token_usage: dict[str, int]
    model: str


@dataclass(frozen=True)
class BriefingResult:
    """Final model-generated briefing payload."""

    content: str
    model: str
    token_usage: dict[str, int]
    generated_at: datetime
