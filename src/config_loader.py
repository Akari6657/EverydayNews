"""Load and validate application configuration from YAML and .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import (
    AppConfig,
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


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


_MISSING = object()


def get_config(
    config_path: str | Path = "config.yaml",
    env_path: str | Path | None = None,
) -> AppConfig:
    """Load application configuration from disk."""

    config_file = Path(config_path).resolve()
    root_dir = config_file.parent
    env_file = Path(env_path).resolve() if env_path else root_dir / ".env"
    _load_environment(env_file)
    payload = _load_yaml(config_file)
    pipeline = _parse_pipeline(payload)
    return AppConfig(
        sources=_parse_sources(payload),
        pipeline=pipeline,
        dedup=_parse_dedup(payload),
        summarizer=_parse_summarizer(payload),
        llm=_parse_llm(payload),
        output=_parse_output(payload),
        schedule=_parse_schedule(payload),
        root_dir=root_dir,
        config_path=config_file,
        thread_clustering=_parse_thread_clustering(payload),
        ranking=_parse_ranking(payload),
    )


def _load_environment(env_path: Path) -> None:
    """Load .env values into the current process."""

    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        _set_env_from_line(line)


def _set_env_from_line(line: str) -> None:
    """Parse a KEY=VALUE line when python-dotenv is unavailable."""

    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return
    key, value = stripped.split("=", 1)
    clean_value = value.strip().strip('"').strip("'")
    os.environ.setdefault(key.strip(), clean_value)


def _load_yaml(config_path: Path) -> dict[str, Any]:
    """Load YAML as a dictionary."""

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError("pyyaml is required to load config.yaml") from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError("Top-level config must be a mapping")
    return payload


def _parse_sources(payload: dict[str, Any]) -> list[SourceConfig]:
    """Parse source definitions."""

    raw_sources = _mapping_list(payload, "sources")
    sources = [_parse_source(source) for source in raw_sources]
    if not sources:
        raise ConfigError("At least one source must be configured")
    return sources


def _parse_source(raw_source: dict[str, Any]) -> SourceConfig:
    """Parse a single source entry."""

    name = _string(raw_source, "name")
    slug = _string(raw_source, "slug")
    raw_feeds = _mapping_list(raw_source, "feeds")
    if not raw_feeds:
        raise ConfigError(f"Source '{slug}' must define at least one feed")
    feeds = [
        FeedConfig(
            url=_string(feed, "url"),
            category=_string(feed, "category"),
            exclude_keywords=_string_list(feed, "exclude_keywords"),
            exclude_categories=_string_list(feed, "exclude_categories"),
        )
        for feed in raw_feeds
    ]
    return SourceConfig(name=name, slug=slug, feeds=feeds)


def _parse_pipeline(payload: dict[str, Any]) -> PipelineConfig:
    """Parse pipeline settings."""

    section = _mapping(payload, "pipeline")
    return PipelineConfig(
        max_articles_per_source=_int(section, "max_articles_per_source"),
        total_articles_for_summary=_int(section, "total_articles_for_summary"),
        importance_threshold=_int(section, "importance_threshold", default=4, minimum=0, maximum=10),
        max_items_per_topic=_int(section, "max_items_per_topic", default=4, positive=True),
        exclude_summary_keywords=_string_list(section, "exclude_summary_keywords"),
        language=_string(section, "language"),
        briefing_style=_string(section, "briefing_style"),
    )


def _parse_dedup(payload: dict[str, Any]) -> DedupConfig:
    """Parse within-thread near-duplicate settings with sensible defaults."""

    section = _mapping(payload, "dedup", required=False)
    if not section:
        return DedupConfig(
            method="embedding",
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            cache_embeddings=True,
            within_thread_similarity_threshold=0.88,
        )
    method = _string(section, "method", "embedding")
    if method not in {"embedding", "difflib"}:
        raise ConfigError("'dedup.method' must be either 'embedding' or 'difflib'")
    within_thread = _mapping(section, "within_thread", required=False)
    return DedupConfig(
        method=method,
        model=_string(section, "model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        cache_embeddings=_bool(section, "cache_embeddings", True),
        within_thread_similarity_threshold=_float(within_thread, "similarity_threshold", 0.88),
    )


def _parse_summarizer(payload: dict[str, Any]) -> SummarizerConfig:
    """Parse map-reduce summarizer settings with defaults."""

    section = _mapping(payload, "summarizer", required=False)
    map_section = _mapping(section, "map", required=False)
    reduce_section = _mapping(section, "reduce", required=False)
    map_config = SummarizerMapConfig(
        batch_size=_int(map_section, "batch_size", default=5, positive=True),
        max_retries=_int(map_section, "max_retries", default=2, positive=True),
    )
    reduce_config = SummarizerReduceConfig(
        top_k=_int(reduce_section, "top_k", default=30, positive=True),
        max_retries=_int(reduce_section, "max_retries", default=2, positive=True),
    )
    return SummarizerConfig(map=map_config, reduce=reduce_config)


def _parse_llm(payload: dict[str, Any]) -> LLMConfig:
    """Parse LLM settings."""

    section = _mapping(payload, "llm")
    provider = _string(section, "provider")
    if provider != "deepseek":
        raise ConfigError("Only the 'deepseek' provider is currently supported")
    return LLMConfig(
        provider=provider,
        model=_string(section, "model"),
        base_url=_string(section, "base_url"),
        api_key_env=_string(section, "api_key_env"),
        max_tokens=_int(section, "max_tokens"),
        temperature=_float(section, "temperature"),
    )


def _parse_thread_clustering(payload: dict[str, Any]) -> ThreadClusteringConfig:
    """Parse optional story-thread clustering settings."""

    llm_section = _mapping(payload, "llm", required=False)
    default_provider = _string(llm_section, "provider", "deepseek")
    default_model = _string(llm_section, "model", "deepseek-chat")
    section = _mapping(payload, "thread_clustering", required=False)
    if not section:
        return ThreadClusteringConfig(
            provider=default_provider,
            model=default_model,
            max_retries=2,
            max_articles_per_call=150,
            max_articles_per_thread=12,
            max_refinement_rounds=1,
        )
    provider = _string(section, "provider", default_provider)
    if provider != "deepseek":
        raise ConfigError("Only the 'deepseek' provider is currently supported for thread clustering")
    return ThreadClusteringConfig(
        provider=provider,
        model=_string(section, "model", default_model),
        max_retries=_int(section, "max_retries", default=2, positive=True),
        max_articles_per_call=_int(section, "max_articles_per_call", default=150, positive=True),
        max_articles_per_thread=_int(section, "max_articles_per_thread", default=12, positive=True),
        max_refinement_rounds=_int(section, "max_refinement_rounds", default=1, positive=True),
    )


def _parse_ranking(payload: dict[str, Any]) -> RankingConfig:
    """Parse optional thread-ranking settings."""

    section = _mapping(payload, "ranking", required=False)
    if not section:
        return RankingConfig(importance_floor=0.0, keep_major_always=True)
    return RankingConfig(
        importance_floor=_float(section, "importance_floor", 0.15),
        keep_major_always=_bool(section, "keep_major_always", True),
    )


def _parse_output(payload: dict[str, Any]) -> OutputConfig:
    """Parse output channel settings."""

    section = _mapping(payload, "output")
    markdown = _parse_markdown_output(_mapping(section, "markdown"))
    json_output = _parse_json_output(_mapping(section, "json", required=False), markdown)
    email = _parse_email_output(_mapping(section, "email"))
    telegram = _parse_telegram_output(_mapping(section, "telegram"))
    return OutputConfig(markdown=markdown, json=json_output, email=email, telegram=telegram)


def _parse_markdown_output(section: dict[str, Any]) -> MarkdownOutputConfig:
    """Parse markdown output settings."""

    return MarkdownOutputConfig(
        enabled=_bool(section, "enabled"),
        directory=_string(section, "directory"),
        group_by_month=_bool(section, "group_by_month", True),
    )


def _parse_json_output(
    section: dict[str, Any],
    markdown: MarkdownOutputConfig,
) -> JsonOutputConfig:
    """Parse JSON output settings, defaulting to a structured sibling directory."""

    if not section:
        return JsonOutputConfig(
            enabled=True,
            directory=_default_json_directory(markdown.directory),
            group_by_month=markdown.group_by_month,
        )
    return JsonOutputConfig(
        enabled=_bool(section, "enabled", True),
        directory=_string(section, "directory", _default_json_directory(markdown.directory)),
        group_by_month=_bool(section, "group_by_month", markdown.group_by_month),
    )


def _parse_email_output(section: dict[str, Any]) -> EmailOutputConfig:
    """Parse email output settings."""

    recipients = _string_list(section, "recipients")
    return EmailOutputConfig(
        enabled=_bool(section, "enabled"),
        smtp_host=str(section.get("smtp_host", "")),
        smtp_port=_int(section, "smtp_port"),
        sender=str(section.get("sender", "")),
        recipients=recipients,
    )


def _parse_telegram_output(section: dict[str, Any]) -> TelegramOutputConfig:
    """Parse Telegram output settings."""

    return TelegramOutputConfig(
        enabled=_bool(section, "enabled"),
        bot_token_env=_string(section, "bot_token_env"),
        chat_id_env=_string(section, "chat_id_env"),
    )


def _parse_schedule(payload: dict[str, Any]) -> ScheduleConfig:
    """Parse schedule settings."""

    section = _mapping(payload, "schedule")
    return ScheduleConfig(
        timezone=_string(section, "timezone"),
        run_at=_string(section, "run_at"),
    )


def _mapping(payload: dict[str, Any], key: str, required: bool = True) -> dict[str, Any]:
    """Return a mapping section or raise when the value is invalid."""

    value = payload.get(key, _MISSING)
    if value is _MISSING or value is None:
        if required:
            raise ConfigError(f"'{key}' must be a mapping")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _mapping_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return a list of mapping items or raise when invalid."""

    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ConfigError(f"'{key}' must be a list of mappings")
    return value


def _string(payload: dict[str, Any], key: str, default: str | object = _MISSING) -> str:
    """Return a non-empty string field, optionally using a default."""

    value = payload.get(key, default)
    if value is _MISSING or not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string")
    return value.strip()


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    """Return a list of non-empty strings."""

    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"'{key}' must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _int(
    payload: dict[str, Any],
    key: str,
    default: int | object = _MISSING,
    *,
    positive: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return a validated integer field."""

    value = payload.get(key, default)
    if value is _MISSING or isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer")
    if positive and value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ConfigError(f"'{key}' must be between {minimum} and {maximum}")
    return value


def _float(payload: dict[str, Any], key: str, default: float | object = _MISSING) -> float:
    """Return a float-compatible field."""

    value = payload.get(key, default)
    if value is _MISSING or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be a number")
    return float(value)


def _bool(payload: dict[str, Any], key: str, default: bool | object = _MISSING) -> bool:
    """Return a boolean field."""

    value = payload.get(key, default)
    if value is _MISSING or not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a boolean")
    return value


def _default_json_directory(markdown_directory: str) -> str:
    """Return a sensible default JSON directory derived from the Markdown directory."""

    if markdown_directory.endswith("/md"):
        return markdown_directory[: -len("/md")] + "/json"
    if markdown_directory == "output":
        return "output/json"
    return f"{markdown_directory}/json"
