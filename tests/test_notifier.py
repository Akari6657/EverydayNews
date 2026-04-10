"""Tests for optional notification channels."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from src import notifier
from src.models import ClusterSummary, FinalBriefing


def _make_briefing() -> FinalBriefing:
    """Create a minimal structured briefing for notifier tests."""

    summary = ClusterSummary(
        cluster_id="thread-1",
        topic="国际政治",
        headline_zh="测试标题",
        summary_zh="测试摘要",
        importance=8,
        entities=["实体A"],
        source_names=["New York Times"],
        primary_link="https://example.com/story",
    )
    return FinalBriefing(
        date="2026-04-08",
        overview_zh="今日综述。",
        topics={"国际政治": [summary]},
        total_clusters=1,
        total_sources=1,
        generated_at=datetime.now(timezone.utc),
        token_usage={"input_tokens": 0, "output_tokens": 0},
        model="deepseek-chat",
    )


def test_notify_sends_telegram_in_chunks(sample_config, monkeypatch, tmp_path: Path) -> None:
    """Notifier should split long Telegram messages."""

    telegram_config = replace(sample_config.output.telegram, enabled=True)
    output_config = replace(sample_config.output, telegram=telegram_config)
    config = replace(sample_config, output=output_config)
    output_path = tmp_path / "briefing.md"
    output_path.write_text("A" * 5000, encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    sent_chunks: list[str] = []

    def fake_post(bot_token: str, chat_id: str, text: str) -> None:
        sent_chunks.append(text)

    monkeypatch.setattr(notifier, "_post_telegram_message", fake_post)
    briefing = _make_briefing()

    notifier.notify(output_path, briefing, config)

    assert len(sent_chunks) == 2


def test_notify_swallows_channel_failures(sample_config, monkeypatch, tmp_path: Path) -> None:
    """Notifier should continue when one delivery channel fails."""

    telegram_config = replace(sample_config.output.telegram, enabled=True)
    output_config = replace(sample_config.output, telegram=telegram_config)
    config = replace(sample_config, output=output_config)
    output_path = tmp_path / "briefing.md"
    output_path.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setattr(notifier, "_post_telegram_message", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    briefing = _make_briefing()

    notifier.notify(output_path, briefing, config)
