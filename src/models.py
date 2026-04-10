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
    language: str
    briefing_style: str


@dataclass(frozen=True)
class DedupConfig:
    """Within-thread near-duplicate cleanup settings."""

    method: str
    model: str
    cache_embeddings: bool
    within_thread_enabled: bool = False
    within_thread_similarity_threshold: float = 0.88


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
class EvaluationConfig:
    """Optional evaluation-stage settings."""

    enabled: bool
    max_retries: int


@dataclass(frozen=True)
class ThreadClusteringConfig:
    """LLM-based story-thread clustering settings."""

    enabled: bool
    provider: str
    model: str
    max_retries: int
    max_articles_per_call: int
    max_articles_per_thread: int
    max_refinement_rounds: int


@dataclass(frozen=True)
class RankingConfig:
    """Thread-ranking settings for the experimental V2 pipeline."""

    importance_floor: float
    keep_major_always: bool


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
    thread_clustering: ThreadClusteringConfig = field(
        default_factory=lambda: ThreadClusteringConfig(
            enabled=True,
            provider="deepseek",
            model="deepseek-chat",
            max_retries=2,
            max_articles_per_call=150,
            max_articles_per_thread=12,
            max_refinement_rounds=1,
        )
    )
    ranking: RankingConfig = field(
        default_factory=lambda: RankingConfig(
            importance_floor=0.15,
            keep_major_always=True,
        )
    )
    evaluation: EvaluationConfig = field(
        default_factory=lambda: EvaluationConfig(enabled=False, max_retries=2)
    )

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
class StoryThread:
    """A group of articles reporting on the same ongoing story."""

    thread_id: int
    topic: str
    topic_en: str
    articles: list[Article]
    source_names: list[str]
    source_count: int
    primary: Article
    latest_published: datetime
    rationale: str = ""

    @property
    def is_multi_source(self) -> bool:
        """Return whether the thread contains multiple sources."""

        return self.source_count >= 2

    @property
    def is_major(self) -> bool:
        """Return whether the thread spans at least three distinct sources."""

        return self.source_count >= 3


@dataclass(frozen=True)
class WithinThreadMerge:
    """One article merged into a stricter within-thread near-duplicate group."""

    kept_article: Article
    removed_article: Article
    similarity: float


@dataclass(frozen=True)
class ThreadDedupDiagnostics:
    """Debug information for within-thread near-duplicate cleanup."""

    before_articles: int
    after_articles: int
    merged_pairs: list[WithinThreadMerge] = field(default_factory=list)


@dataclass(frozen=True)
class ThreadSummary:
    """Map-stage summary for a single story thread."""

    thread_id: str
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

    summaries: list[ThreadSummary]
    token_usage: dict[str, int]
    model: str
    batches_total: int = 0
    batches_failed: int = 0
    threads_skipped: int = 0


@dataclass(frozen=True)
class FinalBriefing:
    """Reduce-stage structured daily briefing."""

    date: str
    overview_zh: str
    topics: dict[str, list[ThreadSummary]]
    total_threads: int
    total_sources: int
    generated_at: datetime
    token_usage: dict[str, int]
    model: str


@dataclass(frozen=True)
class EvaluationResult:
    """LLM-based quality evaluation for one generated briefing."""

    coverage: int
    diversity: int
    clarity: int
    redundancy: int
    importance_calibration: int
    notes: str
    token_usage: dict[str, int]
    model: str
    generated_at: datetime

    @property
    def scores(self) -> dict[str, int]:
        """Return the numeric evaluation scores as a plain dictionary."""

        return {
            "coverage": self.coverage,
            "diversity": self.diversity,
            "clarity": self.clarity,
            "redundancy": self.redundancy,
            "importance_calibration": self.importance_calibration,
        }


@dataclass(frozen=True)
class RunMetrics:
    """Per-run metrics written to the append-only metrics log."""

    date: str
    articles_fetched: int
    threads: int
    map_summaries_generated: int
    after_importance_filter: int
    final_items: int
    total_tokens: int
    duration_seconds: float
    map_batches_total: int
    map_batches_failed: int
    map_threads_skipped: int
    eval_scores: dict[str, int] | None
    eval_notes: str | None
    status: str = "success"
    failure_stage: str | None = None
    failure_reason: str | None = None
