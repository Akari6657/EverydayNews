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
  dedup_similarity_threshold: 0.7
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
    assert config.summarizer.map.batch_size == 5
    assert config.llm.model == "deepseek-chat"
    assert config.pipeline.importance_threshold == 4
    assert config.output.markdown.group_by_month is True
    assert config.output.json.directory == "output/json"
    assert config.output.json.group_by_month is True
    assert config.root_dir == tmp_path


def test_get_config_rejects_missing_sources(tmp_path: Path) -> None:
    """Config loader should fail on invalid structure."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("sources: []\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        get_config(config_path=config_path, env_path=tmp_path / ".env")
