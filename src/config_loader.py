"""Load and validate application configuration from YAML and .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import (
    AppConfig,
    DedupConfig,
    EmailOutputConfig,
    EvaluationConfig,
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


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


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
        dedup=_parse_dedup(payload, pipeline),
        summarizer=_parse_summarizer(payload),
        llm=_parse_llm(payload),
        output=_parse_output(payload),
        schedule=_parse_schedule(payload),
        root_dir=root_dir,
        config_path=config_file,
        evaluation=_parse_evaluation(payload),
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

    raw_sources = _require_mapping_list(payload, "sources")
    sources = [_parse_source(source) for source in raw_sources]
    if not sources:
        raise ConfigError("At least one source must be configured")
    return sources


def _parse_source(raw_source: dict[str, Any]) -> SourceConfig:
    """Parse a single source entry."""

    name = _require_string(raw_source, "name")
    slug = _require_string(raw_source, "slug")
    raw_feeds = _require_mapping_list(raw_source, "feeds")
    if not raw_feeds:
        raise ConfigError(f"Source '{slug}' must define at least one feed")
    feeds = [
        FeedConfig(
            url=_require_string(feed, "url"),
            category=_require_string(feed, "category"),
            exclude_keywords=_optional_string_list(feed, "exclude_keywords"),
            exclude_categories=_optional_string_list(feed, "exclude_categories"),
        )
        for feed in raw_feeds
    ]
    return SourceConfig(name=name, slug=slug, feeds=feeds)


def _parse_pipeline(payload: dict[str, Any]) -> PipelineConfig:
    """Parse pipeline settings."""

    section = _require_mapping(payload, "pipeline")
    return PipelineConfig(
        max_articles_per_source=_require_int(section, "max_articles_per_source"),
        total_articles_for_summary=_require_int(section, "total_articles_for_summary"),
        importance_threshold=_bounded_int_with_default(
            section,
            "importance_threshold",
            default=4,
            minimum=0,
            maximum=10,
        ),
        max_items_per_topic=_positive_int_with_default(section, "max_items_per_topic", 4),
        exclude_summary_keywords=_optional_string_list(section, "exclude_summary_keywords"),
        dedup_similarity_threshold=_require_float(section, "dedup_similarity_threshold"),
        language=_require_string(section, "language"),
        briefing_style=_require_string(section, "briefing_style"),
    )


def _parse_dedup(payload: dict[str, Any], pipeline: PipelineConfig) -> DedupConfig:
    """Parse deduplication settings with sensible defaults."""

    section = payload.get("dedup")
    if section is None:
        return DedupConfig(
            method="embedding",
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            similarity_threshold=pipeline.dedup_similarity_threshold,
            clustering_algorithm="greedy",
            cache_embeddings=True,
        )
    if not isinstance(section, dict):
        raise ConfigError("'dedup' must be a mapping")
    method = str(section.get("method", "embedding")).strip()
    clustering_algorithm = str(section.get("clustering_algorithm", "greedy")).strip()
    if method not in {"embedding", "difflib"}:
        raise ConfigError("'dedup.method' must be either 'embedding' or 'difflib'")
    if clustering_algorithm not in {"greedy", "dbscan"}:
        raise ConfigError("'dedup.clustering_algorithm' must be either 'greedy' or 'dbscan'")
    cache_embeddings = section.get("cache_embeddings", True)
    if not isinstance(cache_embeddings, bool):
        raise ConfigError("'dedup.cache_embeddings' must be a boolean")
    return DedupConfig(
        method=method,
        model=str(
            section.get(
                "model",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        ).strip(),
        similarity_threshold=float(section.get("similarity_threshold", pipeline.dedup_similarity_threshold)),
        clustering_algorithm=clustering_algorithm,
        cache_embeddings=cache_embeddings,
    )


def _parse_summarizer(payload: dict[str, Any]) -> SummarizerConfig:
    """Parse map-reduce summarizer settings with defaults."""

    section = payload.get("summarizer", {})
    if not isinstance(section, dict):
        raise ConfigError("'summarizer' must be a mapping")
    map_section = section.get("map", {})
    reduce_section = section.get("reduce", {})
    if not isinstance(map_section, dict):
        raise ConfigError("'summarizer.map' must be a mapping")
    if not isinstance(reduce_section, dict):
        raise ConfigError("'summarizer.reduce' must be a mapping")
    map_config = SummarizerMapConfig(
        batch_size=_positive_int(map_section.get("batch_size", 5), "summarizer.map.batch_size"),
        max_retries=_positive_int(map_section.get("max_retries", 2), "summarizer.map.max_retries"),
    )
    reduce_config = SummarizerReduceConfig(
        top_k=_positive_int(reduce_section.get("top_k", 30), "summarizer.reduce.top_k"),
        max_retries=_positive_int(reduce_section.get("max_retries", 2), "summarizer.reduce.max_retries"),
    )
    return SummarizerConfig(map=map_config, reduce=reduce_config)


def _parse_llm(payload: dict[str, Any]) -> LLMConfig:
    """Parse LLM settings."""

    section = _require_mapping(payload, "llm")
    provider = _require_string(section, "provider")
    if provider != "deepseek":
        raise ConfigError("Only the 'deepseek' provider is currently supported")
    return LLMConfig(
        provider=provider,
        model=_require_string(section, "model"),
        base_url=_require_string(section, "base_url"),
        api_key_env=_require_string(section, "api_key_env"),
        max_tokens=_require_int(section, "max_tokens"),
        temperature=_require_float(section, "temperature"),
    )


def _parse_output(payload: dict[str, Any]) -> OutputConfig:
    """Parse output channel settings."""

    section = _require_mapping(payload, "output")
    markdown = _parse_markdown_output(_require_mapping(section, "markdown"))
    json_output = _parse_json_output(section.get("json"), markdown)
    email = _parse_email_output(_require_mapping(section, "email"))
    telegram = _parse_telegram_output(_require_mapping(section, "telegram"))
    return OutputConfig(markdown=markdown, json=json_output, email=email, telegram=telegram)


def _parse_markdown_output(section: dict[str, Any]) -> MarkdownOutputConfig:
    """Parse markdown output settings."""

    return MarkdownOutputConfig(
        enabled=_require_bool(section, "enabled"),
        directory=_require_string(section, "directory"),
        group_by_month=_require_bool_with_default(section, "group_by_month", True),
    )


def _parse_json_output(
    section: Any,
    markdown: MarkdownOutputConfig,
) -> JsonOutputConfig:
    """Parse JSON output settings, defaulting to a structured sibling directory."""

    if section is None:
        return JsonOutputConfig(
            enabled=True,
            directory=_default_json_directory(markdown.directory),
            group_by_month=markdown.group_by_month,
        )
    if not isinstance(section, dict):
        raise ConfigError("'output.json' must be a mapping")
    return JsonOutputConfig(
        enabled=_require_bool_with_default(section, "enabled", True),
        directory=_require_string_with_default(
            section,
            "directory",
            _default_json_directory(markdown.directory),
        ),
        group_by_month=_require_bool_with_default(section, "group_by_month", markdown.group_by_month),
    )


def _parse_email_output(section: dict[str, Any]) -> EmailOutputConfig:
    """Parse email output settings."""

    recipients = _require_string_list(section, "recipients")
    return EmailOutputConfig(
        enabled=_require_bool(section, "enabled"),
        smtp_host=str(section.get("smtp_host", "")),
        smtp_port=_require_int(section, "smtp_port"),
        sender=str(section.get("sender", "")),
        recipients=recipients,
    )


def _parse_telegram_output(section: dict[str, Any]) -> TelegramOutputConfig:
    """Parse Telegram output settings."""

    return TelegramOutputConfig(
        enabled=_require_bool(section, "enabled"),
        bot_token_env=_require_string(section, "bot_token_env"),
        chat_id_env=_require_string(section, "chat_id_env"),
    )


def _parse_schedule(payload: dict[str, Any]) -> ScheduleConfig:
    """Parse schedule settings."""

    section = _require_mapping(payload, "schedule")
    return ScheduleConfig(
        timezone=_require_string(section, "timezone"),
        run_at=_require_string(section, "run_at"),
    )


def _parse_evaluation(payload: dict[str, Any]) -> EvaluationConfig:
    """Parse optional evaluation settings with conservative defaults."""

    section = payload.get("evaluation")
    if section is None:
        return EvaluationConfig(enabled=False, max_retries=2)
    if not isinstance(section, dict):
        raise ConfigError("'evaluation' must be a mapping")
    return EvaluationConfig(
        enabled=_require_bool_with_default(section, "enabled", False),
        max_retries=_positive_int(section.get("max_retries", 2), "evaluation.max_retries"),
    )


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Require a dictionary at the given key."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    return value


def _require_mapping_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Require a list of dictionaries at the given key."""

    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ConfigError(f"'{key}' must be a list of mappings")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    """Require a non-empty string."""

    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string")
    return value.strip()


def _require_string_with_default(payload: dict[str, Any], key: str, default: str) -> str:
    """Require a string when present, otherwise use a default."""

    if key not in payload:
        return default
    return _require_string(payload, key)


def _require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    """Require a list of strings."""

    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"'{key}' must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _optional_string_list(payload: dict[str, Any], key: str) -> list[str]:
    """Return an optional list of strings, defaulting to an empty list."""

    if key not in payload:
        return []
    return _require_string_list(payload, key)


def _require_int(payload: dict[str, Any], key: str) -> int:
    """Require an integer value."""

    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    """Require a float-compatible value."""

    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be a number")
    return float(value)


def _require_bool(payload: dict[str, Any], key: str) -> bool:
    """Require a boolean value."""

    value = payload.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be a boolean")
    return value


def _require_bool_with_default(payload: dict[str, Any], key: str, default: bool) -> bool:
    """Require a boolean when present, otherwise use a default."""

    if key not in payload:
        return default
    return _require_bool(payload, key)


def _positive_int(value: Any, key: str) -> int:
    """Require a positive integer value."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer")
    return value


def _positive_int_with_default(payload: dict[str, Any], key: str, default: int) -> int:
    """Require a positive integer when present, otherwise use a default."""

    if key not in payload:
        return default
    return _positive_int(payload.get(key), key)


def _bounded_int_with_default(
    payload: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Require an integer within a closed range when present, else use a default."""

    if key not in payload:
        return default
    value = _require_int(payload, key)
    if value < minimum or value > maximum:
        raise ConfigError(f"'{key}' must be between {minimum} and {maximum}")
    return value


def _default_json_directory(markdown_directory: str) -> str:
    """Return a sensible default JSON directory derived from the Markdown directory."""

    if markdown_directory.endswith("/md"):
        return markdown_directory[: -len("/md")] + "/json"
    if markdown_directory == "output":
        return "output/json"
    return f"{markdown_directory}/json"
