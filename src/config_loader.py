"""Load and validate application configuration from YAML and .env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import (
    AppConfig,
    EmailOutputConfig,
    FeedConfig,
    LLMConfig,
    MarkdownOutputConfig,
    OutputConfig,
    PipelineConfig,
    ScheduleConfig,
    SourceConfig,
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
    return AppConfig(
        sources=_parse_sources(payload),
        pipeline=_parse_pipeline(payload),
        llm=_parse_llm(payload),
        output=_parse_output(payload),
        schedule=_parse_schedule(payload),
        root_dir=root_dir,
        config_path=config_file,
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
    feeds = [FeedConfig(_require_string(feed, "url"), _require_string(feed, "category")) for feed in raw_feeds]
    return SourceConfig(name=name, slug=slug, feeds=feeds)


def _parse_pipeline(payload: dict[str, Any]) -> PipelineConfig:
    """Parse pipeline settings."""

    section = _require_mapping(payload, "pipeline")
    return PipelineConfig(
        max_articles_per_source=_require_int(section, "max_articles_per_source"),
        total_articles_for_summary=_require_int(section, "total_articles_for_summary"),
        dedup_similarity_threshold=_require_float(section, "dedup_similarity_threshold"),
        language=_require_string(section, "language"),
        briefing_style=_require_string(section, "briefing_style"),
    )


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
    email = _parse_email_output(_require_mapping(section, "email"))
    telegram = _parse_telegram_output(_require_mapping(section, "telegram"))
    return OutputConfig(markdown=markdown, email=email, telegram=telegram)


def _parse_markdown_output(section: dict[str, Any]) -> MarkdownOutputConfig:
    """Parse markdown output settings."""

    return MarkdownOutputConfig(
        enabled=_require_bool(section, "enabled"),
        directory=_require_string(section, "directory"),
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


def _require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    """Require a list of strings."""

    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"'{key}' must be a list of strings")
    return [item.strip() for item in value if item.strip()]


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
