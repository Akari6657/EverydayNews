"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import ConfigError, get_config


BASE_CONFIG = """
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
    directory: {markdown_directory}
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
"""


def _write_config(tmp_path: Path, extra: str = "", markdown_directory: str = "output/md") -> Path:
    """Write a minimal valid config with optional extra YAML appended."""

    config_path = tmp_path / "config.yaml"
    body = BASE_CONFIG.format(markdown_directory=markdown_directory).strip()
    if extra.strip():
        body = f"{body}\n{extra.strip()}\n"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_get_config_loads_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Config loader should parse YAML and .env files."""

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config_path = _write_config(tmp_path, markdown_directory="output")
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")

    config = get_config(config_path=config_path, env_path=env_path)

    assert config.sources[0].slug == "example"
    assert config.dedup.method == "embedding"
    assert config.dedup.within_thread_similarity_threshold == 0.88
    assert config.summarizer.map.batch_size == 5
    assert config.llm.model == "deepseek-chat"
    assert config.pipeline.importance_threshold == 4
    assert config.pipeline.max_items_per_topic == 4
    assert config.pipeline.exclude_summary_keywords == []
    assert config.output.markdown.group_by_month is True
    assert config.output.json.directory == "output/json"
    assert config.output.json.group_by_month is True
    assert config.thread_clustering.max_articles_per_call == 150
    assert config.thread_clustering.max_articles_per_thread == 12
    assert config.thread_clustering.max_refinement_rounds == 1
    assert config.ranking.importance_floor == 0.15
    assert config.ranking.keep_major_always is True
    assert config.root_dir == tmp_path


def test_get_config_rejects_missing_sources(tmp_path: Path) -> None:
    """Config loader should fail on invalid structure."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text("sources: []\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        get_config(config_path=config_path, env_path=tmp_path / ".env")


@pytest.mark.parametrize(
    ("extra", "assertions"),
    [
        (
            """
thread_clustering:
  provider: deepseek
  model: deepseek-chat
  max_retries: 3
  max_articles_per_call: 120
  max_articles_per_thread: 10
  max_refinement_rounds: 2
""",
            lambda config: (
                config.thread_clustering.max_retries == 3
                and config.thread_clustering.max_articles_per_call == 120
                and config.thread_clustering.max_articles_per_thread == 10
                and config.thread_clustering.max_refinement_rounds == 2
            ),
        ),
        (
            """
dedup:
  method: embedding
  within_thread:
    similarity_threshold: 0.91
""",
            lambda config: config.dedup.within_thread_similarity_threshold == 0.91,
        ),
        (
            """
ranking:
  importance_floor: 0.25
  keep_major_always: false
""",
            lambda config: (
                config.ranking.importance_floor == 0.25
                and config.ranking.keep_major_always is False
            ),
        ),
    ],
)
def test_get_config_parses_optional_sections(
    tmp_path: Path,
    extra: str,
    assertions,
) -> None:
    """Config loader should parse optional tuning sections."""

    config = get_config(config_path=_write_config(tmp_path, extra=extra), env_path=tmp_path / ".env")

    assert assertions(config)
