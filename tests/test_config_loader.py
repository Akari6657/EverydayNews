"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import ConfigError, get_config


def test_get_config_loads_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config loader should parse YAML and .env files."""

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(
        """
sources:
  - name: Example
    slug: example
    feeds:
      - url: https://example.com/rss.xml
        category: top
pipeline:
  max_articles_per_source: 10
  total_articles_for_summary: 20
  language: zh-CN
  briefing_style: concise
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 4096
  temperature: 0.3
output:
  markdown:
    enabled: true
    directory: output
  email:
    enabled: false
    smtp_host: ""
    smtp_port: 587
    sender: ""
    recipients: []
  telegram:
    enabled: false
    bot_token_env: TELEGRAM_BOT_TOKEN
    chat_id_env: TELEGRAM_CHAT_ID
schedule:
  timezone: Asia/Shanghai
  run_at: "08:00"
""".strip(),
        encoding="utf-8",
    )
    env_path.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")

    config = get_config(config_path=config_path, env_path=env_path)

    assert config.sources[0].slug == "example"
    assert config.dedup.method == "embedding"
    assert config.dedup.within_thread_enabled is False
    assert config.dedup.within_thread_similarity_threshold == 0.88
    assert config.summarizer.map.batch_size == 5
    assert config.llm.model == "deepseek-chat"
    assert config.pipeline.importance_threshold == 4
    assert config.pipeline.max_items_per_topic == 4
    assert config.pipeline.exclude_summary_keywords == []
    assert config.output.markdown.group_by_month is True
    assert config.output.json.directory == "output/json"
    assert config.output.json.group_by_month is True
    assert config.thread_clustering.enabled is True
    assert config.thread_clustering.max_articles_per_call == 150
    assert config.thread_clustering.max_articles_per_thread == 12
    assert config.thread_clustering.max_refinement_rounds == 1
    assert config.ranking.importance_floor == 0.15
    assert config.ranking.keep_major_always is True
    assert config.evaluation.enabled is False
    assert config.evaluation.max_retries == 2
    assert config.root_dir == tmp_path


def test_get_config_rejects_missing_sources(tmp_path: Path) -> None:
    """Config loader should fail on invalid structure."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("sources: []\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        get_config(config_path=config_path, env_path=tmp_path / ".env")


def test_get_config_parses_evaluation_section(tmp_path: Path) -> None:
    """Config loader should parse optional evaluation settings."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
sources:
  - name: Example
    slug: example
    feeds:
      - url: https://example.com/rss.xml
        category: top
pipeline:
  max_articles_per_source: 10
  total_articles_for_summary: 20
  language: zh-CN
  briefing_style: concise
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 4096
  temperature: 0.3
output:
  markdown:
    enabled: true
    directory: output/md
  email:
    enabled: false
    smtp_host: ""
    smtp_port: 587
    sender: ""
    recipients: []
  telegram:
    enabled: false
    bot_token_env: TELEGRAM_BOT_TOKEN
    chat_id_env: TELEGRAM_CHAT_ID
schedule:
  timezone: Asia/Shanghai
  run_at: "08:00"
evaluation:
  enabled: true
  max_retries: 3
""".strip(),
        encoding="utf-8",
    )

    config = get_config(config_path=config_path, env_path=tmp_path / ".env")

    assert config.evaluation.enabled is True
    assert config.evaluation.max_retries == 3


def test_get_config_parses_thread_clustering_section(tmp_path: Path) -> None:
    """Config loader should parse optional thread clustering settings."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
sources:
  - name: Example
    slug: example
    feeds:
      - url: https://example.com/rss.xml
        category: top
pipeline:
  max_articles_per_source: 10
  total_articles_for_summary: 20
  language: zh-CN
  briefing_style: concise
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 4096
  temperature: 0.3
output:
  markdown:
    enabled: true
    directory: output/md
  email:
    enabled: false
    smtp_host: ""
    smtp_port: 587
    sender: ""
    recipients: []
  telegram:
    enabled: false
    bot_token_env: TELEGRAM_BOT_TOKEN
    chat_id_env: TELEGRAM_CHAT_ID
schedule:
  timezone: Asia/Shanghai
  run_at: "08:00"
thread_clustering:
  enabled: true
  provider: deepseek
  model: deepseek-chat
  max_retries: 3
  max_articles_per_call: 120
  max_articles_per_thread: 10
  max_refinement_rounds: 2
""".strip(),
        encoding="utf-8",
    )

    config = get_config(config_path=config_path, env_path=tmp_path / ".env")

    assert config.thread_clustering.enabled is True
    assert config.thread_clustering.max_retries == 3
    assert config.thread_clustering.max_articles_per_call == 120
    assert config.thread_clustering.max_articles_per_thread == 10
    assert config.thread_clustering.max_refinement_rounds == 2


def test_get_config_parses_within_thread_dedup_section(tmp_path: Path) -> None:
    """Config loader should parse optional within-thread dedup settings."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
sources:
  - name: Example
    slug: example
    feeds:
      - url: https://example.com/rss.xml
        category: top
pipeline:
  max_articles_per_source: 10
  total_articles_for_summary: 20
  language: zh-CN
  briefing_style: concise
dedup:
  method: embedding
  within_thread:
    enabled: true
    similarity_threshold: 0.91
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 4096
  temperature: 0.3
output:
  markdown:
    enabled: true
    directory: output/md
  email:
    enabled: false
    smtp_host: ""
    smtp_port: 587
    sender: ""
    recipients: []
  telegram:
    enabled: false
    bot_token_env: TELEGRAM_BOT_TOKEN
    chat_id_env: TELEGRAM_CHAT_ID
schedule:
  timezone: Asia/Shanghai
  run_at: "08:00"
""".strip(),
        encoding="utf-8",
    )

    config = get_config(config_path=config_path, env_path=tmp_path / ".env")

    assert config.dedup.within_thread_enabled is True
    assert config.dedup.within_thread_similarity_threshold == 0.91


def test_get_config_parses_ranking_section(tmp_path: Path) -> None:
    """Config loader should parse optional ranking settings."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
sources:
  - name: Example
    slug: example
    feeds:
      - url: https://example.com/rss.xml
        category: top
pipeline:
  max_articles_per_source: 10
  total_articles_for_summary: 20
  language: zh-CN
  briefing_style: concise
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  max_tokens: 4096
  temperature: 0.3
output:
  markdown:
    enabled: true
    directory: output/md
  email:
    enabled: false
    smtp_host: ""
    smtp_port: 587
    sender: ""
    recipients: []
  telegram:
    enabled: false
    bot_token_env: TELEGRAM_BOT_TOKEN
    chat_id_env: TELEGRAM_CHAT_ID
schedule:
  timezone: Asia/Shanghai
  run_at: "08:00"
ranking:
  importance_floor: 0.25
  keep_major_always: false
""".strip(),
        encoding="utf-8",
    )

    config = get_config(config_path=config_path, env_path=tmp_path / ".env")

    assert config.ranking.importance_floor == 0.25
    assert config.ranking.keep_major_always is False
