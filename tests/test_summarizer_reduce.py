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


def test_build_final_briefing_groups_topics_and_aggregates_usage(sample_config) -> None:
    """Reduce-stage summarizer should return a structured FinalBriefing."""

    summaries = [
        _summary(
            "cluster-a",
            topic="国际政治",
            headline_zh="伊朗局势升级",
            importance=9,
            source_names=["New York Times", "BBC News"],
        ),
        _summary(
            "cluster-b",
            topic="经济金融",
            headline_zh="油价波动加剧",
            importance=7,
            source_names=["BBC News"],
        ),
    ]
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "overview_zh": "国际局势与能源市场成为今日焦点。",
                            "topics": {
                                "国际政治": ["cluster-a"],
                                "经济金融": ["cluster-b"],
                            },
                        },
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
    assert list(briefing.topics) == ["国际政治", "经济金融"]
    assert briefing.total_clusters == 2
    assert briefing.total_sources == 2
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
                                {
                                    "overview_zh": "今日焦点集中在国际政治。",
                                    "topics": {"国际政治": ["cluster-1"]},
                                },
                                ensure_ascii=False,
                            )
                        )
                    )
                ]
            ),
        ]
    )

    briefing = build_final_briefing([_summary("cluster-1")], sample_config, client=client)

    assert len(client.chat.completions.calls) == 2
    assert list(briefing.topics) == ["国际政治"]
    assert briefing.topics["国际政治"][0].cluster_id == "cluster-1"


def test_build_final_briefing_falls_back_after_content_risk(sample_config) -> None:
    """Content-risk request failures should fall back to deterministic local assembly."""

    summaries = [
        _summary("cluster-1", topic="国际政治", importance=9),
        _summary("cluster-2", topic="经济金融", importance=7),
    ]
    client = FakeClient(
        [
            RuntimeError("Error code: 400 - Content Exists Risk"),
            RuntimeError("Error code: 400 - Content Exists Risk"),
        ]
    )

    briefing = build_final_briefing(summaries, sample_config, client=client)

    assert briefing.model.endswith("(fallback)")
    assert briefing.total_clusters == 2
    assert "今日简报重点涵盖" in briefing.overview_zh
    assert list(briefing.topics) == ["国际政治", "经济金融"]


def test_build_final_briefing_appends_missing_clusters(sample_config) -> None:
    """Omitted clusters should be appended back under their original topics."""

    summaries = [
        _summary("cluster-1", topic="国际政治", importance=9),
        _summary("cluster-2", topic="科技", importance=6),
    ]
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "overview_zh": "今日以国际政治为主。",
                            "topics": {"国际政治": ["cluster-1"]},
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, sample_config, client=client)

    assert "科技" in briefing.topics
    assert briefing.topics["科技"][0].cluster_id == "cluster-2"


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
        _summary("cluster-1", importance=4),
        _summary("cluster-2", importance=9),
    ]
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "overview_zh": "只保留了最高优先级新闻。",
                            "topics": {"国际政治": ["cluster-2"]},
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, config, client=client)

    assert briefing.total_clusters == 1
    assert briefing.topics["国际政治"][0].cluster_id == "cluster-2"


def test_build_final_briefing_filters_below_importance_threshold(sample_config) -> None:
    """Reduce-stage should drop summaries below the configured importance threshold."""

    summaries = [
        _summary("cluster-1", importance=3),
        _summary("cluster-2", importance=4),
        _summary("cluster-3", importance=7),
    ]
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "overview_zh": "保留了更重要的新闻。",
                            "topics": {"国际政治": ["cluster-3", "cluster-2"]},
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, sample_config, client=client)

    assert briefing.total_clusters == 2
    assert [item.cluster_id for item in briefing.topics["国际政治"]] == ["cluster-3", "cluster-2"]


def test_build_final_briefing_filters_noise_keywords(sample_config) -> None:
    """Configured summary noise keywords should exclude low-signal live wrappers."""

    summaries = [
        _summary("cluster-1", headline_zh="伊朗战争最新动态", importance=8),
        _summary("cluster-2", headline_zh="特朗普设定新期限", importance=8),
    ]
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "overview_zh": "保留了更具信息量的新闻。",
                            "topics": {"国际政治": ["cluster-2"]},
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, sample_config, client=client)

    assert briefing.total_clusters == 1
    assert [item.cluster_id for item in briefing.topics["国际政治"]] == ["cluster-2"]


def test_build_final_briefing_trims_each_topic_to_limit(sample_config) -> None:
    """Each topic should keep at most the configured number of items."""

    config = replace(sample_config, pipeline=replace(sample_config.pipeline, max_items_per_topic=2))
    summaries = [
        _summary("cluster-1", importance=9),
        _summary("cluster-2", importance=8),
        _summary("cluster-3", importance=7),
    ]
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "overview_zh": "测试主题上限。",
                            "topics": {"国际政治": ["cluster-1", "cluster-2", "cluster-3"]},
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    briefing = build_final_briefing(summaries, config, client=client)

    assert briefing.total_clusters == 2
    assert [item.cluster_id for item in briefing.topics["国际政治"]] == ["cluster-1", "cluster-2"]


def test_build_final_briefing_falls_back_on_unknown_cluster_id(sample_config) -> None:
    """Unknown cluster ids from the model should trigger local fallback assembly."""

    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "overview_zh": "测试概述。",
                            "topics": {"国际政治": ["unknown"]},
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    briefing = build_final_briefing([_summary("cluster-1")], sample_config, client=client)

    assert briefing.model.endswith("(fallback)")
    assert briefing.total_clusters == 1
    assert briefing.topics["国际政治"][0].cluster_id == "cluster-1"


def test_build_final_briefing_still_accepts_object_items(sample_config) -> None:
    """Reduce-stage parser should remain compatible with object items."""

    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "overview_zh": "测试概述。",
                            "topics": {
                                "国际政治": [
                                    {
                                        "cluster_id": "cluster-1",
                                        "headline_zh": "覆盖后的标题",
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    briefing = build_final_briefing([_summary("cluster-1")], sample_config, client=client)

    assert briefing.topics["国际政治"][0].headline_zh == "覆盖后的标题"


def _summary(
    cluster_id: str,
    topic: str = "国际政治",
    headline_zh: str = "默认标题",
    summary_zh: str = "默认摘要",
    importance: int = 8,
    source_names: list[str] | None = None,
):
    """Build a ClusterSummary test object."""

    from src.models import ClusterSummary

    return ClusterSummary(
        cluster_id=cluster_id,
        topic=topic,
        headline_zh=headline_zh,
        summary_zh=summary_zh,
        importance=importance,
        entities=["实体A"],
        source_names=source_names or ["New York Times"],
        primary_link=f"https://example.com/{cluster_id}",
    )
