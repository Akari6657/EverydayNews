"""Tests for reduce-stage structured briefing generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import pytest

from src.summarizer_reduce import build_final_briefing


@dataclass
class FakeMessage:
    """Fake assistant message."""

    content: str


@dataclass
class FakeChoice:
    """Fake response choice wrapper."""

    message: FakeMessage


@dataclass
class FakeUsage:
    """Fake token usage payload."""

    prompt_tokens: int = 444
    completion_tokens: int = 111


@dataclass
class FakeResponse:
    """Fake OpenAI-compatible response."""

    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "deepseek-chat"


class FakeCompletions:
    """Fake chat completions endpoint."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    """Fake OpenAI client with nested chat endpoints."""

    def __init__(self, responses):
        self.chat = type("Chat", (), {"completions": FakeCompletions(responses)})()


def test_build_final_briefing_partitions_top_and_other_stories(sample_config) -> None:
    """Reduce-stage summarizer should produce a two-layer FinalBriefing."""

    summaries = [
        _summary(
            "thread-a",
            topic="国际政治",
            headline_zh="伊朗局势升级",
            importance=9,
            source_names=["New York Times", "BBC News"],
            source_count=2,
            article_count=3,
        ),
        _summary(
            "thread-b",
            topic="经济金融",
            headline_zh="油价波动加剧",
            importance=5,
            source_names=["BBC News"],
            source_count=1,
            article_count=2,
        ),
    ]
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {"overview_zh": "国际局势与能源市场成为今日焦点。"},
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(
        summaries,
        sample_config,
        client=client,
        now=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 100, "output_tokens": 50},
    )

    assert briefing.overview_zh.startswith("国际局势")
    assert [item.thread_id for item in briefing.top_stories] == ["thread-a"]
    assert [item.thread_id for item in briefing.other_stories] == ["thread-b"]
    assert briefing.total_threads == 2
    assert briefing.total_sources == 2
    assert briefing.total_articles == 5
    assert briefing.token_usage == {"input_tokens": 544, "output_tokens": 161}
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_build_final_briefing_retries_invalid_json(sample_config, monkeypatch) -> None:
    """Invalid JSON should trigger one retry before succeeding."""

    monkeypatch.setattr("src.summarizer_reduce.time.sleep", lambda _: None)
    client = FakeClient(
        [
            FakeResponse(choices=[FakeChoice(FakeMessage("not-json"))]),
            FakeResponse(
                choices=[
                    FakeChoice(
                        FakeMessage(
                            json.dumps(
                                {"overview_zh": "今日焦点集中在国际政治。"},
                                ensure_ascii=False,
                            )
                        )
                    )
                ]
            ),
        ]
    )

    briefing = build_final_briefing([_summary("thread-1")], sample_config, client=client)

    assert len(client.chat.completions.calls) == 2
    assert [item.thread_id for item in briefing.top_stories] == ["thread-1"]


def test_build_final_briefing_falls_back_after_content_risk(sample_config) -> None:
    """Content-risk request failures should fall back to deterministic local assembly."""

    summaries = [
        _summary("thread-1", topic="国际政治", importance=9),
        _summary("thread-2", topic="经济金融", importance=5),
    ]
    client = FakeClient(
        [
            RuntimeError("Error code: 400 - Content Exists Risk"),
            RuntimeError("Error code: 400 - Content Exists Risk"),
        ]
    )

    briefing = build_final_briefing(summaries, sample_config, client=client)

    assert briefing.model.endswith("(fallback)")
    assert "今日简报重点涵盖" in briefing.overview_zh
    assert [item.thread_id for item in briefing.top_stories] == ["thread-1"]
    assert [item.thread_id for item in briefing.other_stories] == ["thread-2"]


def test_build_final_briefing_limits_selected_summaries(sample_config) -> None:
    """Reduce-stage should honor the configured top-k selection."""

    config = replace(
        sample_config,
        summarizer=replace(
            sample_config.summarizer,
            reduce=replace(sample_config.summarizer.reduce, top_k=1),
        ),
    )
    summaries = [
        _summary("thread-1", importance=4),
        _summary("thread-2", importance=9),
    ]
    response = FakeResponse(
        choices=[FakeChoice(FakeMessage(json.dumps({"overview_zh": "只保留了最高优先级新闻。"}, ensure_ascii=False)))]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, config, client=client)

    assert briefing.total_threads == 1
    assert [item.thread_id for item in briefing.top_stories] == ["thread-2"]
    assert briefing.other_stories == []


def test_build_final_briefing_filters_below_importance_threshold(sample_config) -> None:
    """Reduce-stage should drop summaries below the configured importance threshold."""

    summaries = [
        _summary("thread-1", importance=3),
        _summary("thread-2", importance=4),
        _summary("thread-3", importance=7),
    ]
    response = FakeResponse(
        choices=[FakeChoice(FakeMessage(json.dumps({"overview_zh": "保留了更重要的新闻。"}, ensure_ascii=False)))]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, sample_config, client=client)

    assert briefing.total_threads == 2
    assert [item.thread_id for item in briefing.top_stories] == ["thread-3"]
    assert [item.thread_id for item in briefing.other_stories] == ["thread-2"]


def test_build_final_briefing_keeps_multisource_story_even_at_lower_importance(sample_config) -> None:
    """Multi-source stories should still land in the top-stories section."""

    summaries = [
        _summary("thread-1", importance=5, source_count=2, source_names=["BBC News", "NPR"]),
        _summary("thread-2", importance=5, source_count=1, source_names=["BBC News"]),
    ]
    response = FakeResponse(
        choices=[FakeChoice(FakeMessage(json.dumps({"overview_zh": "多源报道仍然值得关注。"}, ensure_ascii=False)))]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, sample_config, client=client)

    assert [item.thread_id for item in briefing.top_stories] == ["thread-1"]
    assert [item.thread_id for item in briefing.other_stories] == ["thread-2"]


def test_build_final_briefing_filters_noise_keywords(sample_config) -> None:
    """Configured summary noise keywords should exclude low-signal wrappers."""

    summaries = [
        _summary("thread-1", headline_zh="伊朗战争最新动态", importance=8),
        _summary("thread-2", headline_zh="特朗普设定新期限", importance=8),
    ]
    response = FakeResponse(
        choices=[FakeChoice(FakeMessage(json.dumps({"overview_zh": "保留了更具信息量的新闻。"}, ensure_ascii=False)))]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, sample_config, client=client)

    assert briefing.total_threads == 1
    assert [item.thread_id for item in briefing.top_stories] == ["thread-2"]


def test_build_final_briefing_returns_empty_briefing_when_no_summary_survives(sample_config) -> None:
    """An empty reduce input should still produce a valid empty briefing."""

    briefing = build_final_briefing([_summary("thread-1", importance=1)], sample_config, client=FakeClient([]))

    assert briefing.overview_zh == "今日暂无新的头条新闻。"
    assert briefing.top_stories == []
    assert briefing.other_stories == []
    assert briefing.total_threads == 0
    assert briefing.total_articles == 0


def _summary(
    thread_id: str,
    topic: str = "国际政治",
    headline_zh: str = "默认标题",
    summary_zh: str = "默认摘要",
    importance: int = 8,
    source_names: list[str] | None = None,
    source_count: int | None = None,
    article_count: int = 1,
):
    """Build a ThreadSummary test object."""

    from src.models import ThreadSummary

    names = source_names or ["New York Times"]
    return ThreadSummary(
        thread_id=thread_id,
        topic=topic,
        headline_zh=headline_zh,
        summary_zh=summary_zh,
        importance=importance,
        entities=["实体A"],
        source_names=names,
        primary_link=f"https://example.com/{thread_id}",
        source_count=source_count if source_count is not None else len(names),
        article_count=article_count,
        all_links=[(name, f"https://example.com/{thread_id}-{index}") for index, name in enumerate(names, start=1)],
    )
