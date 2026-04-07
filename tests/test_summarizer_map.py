"""Tests for map-stage structured cluster summarization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from src.summarizer_map import summarize_clusters, summarize_clusters_with_usage


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

    prompt_tokens: int = 111
    completion_tokens: int = 222


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


def _response_for_batch(batch_size: int) -> FakeResponse:
    """Return one valid JSON response for a batch."""

    items = []
    for index in range(batch_size):
        items.append(
            {
                "topic": "国际政治",
                "headline_zh": f"中文标题 {index + 1}",
                "summary_zh": f"摘要 {index + 1}",
                "importance": 8,
                "entities": ["伊朗", "美国"],
            }
        )
    return FakeResponse(choices=[FakeChoice(FakeMessage(json.dumps({"items": items}, ensure_ascii=False)))])


def test_summarize_clusters_batches_requests(sample_config, make_cluster) -> None:
    """Map-stage summarizer should batch clusters by configured batch size."""

    config = replace(
        sample_config,
        summarizer=replace(
            sample_config.summarizer,
            map=replace(sample_config.summarizer.map, batch_size=5),
        ),
    )
    clusters = [make_cluster(cluster_id=f"cluster-{index}") for index in range(12)]
    client = FakeClient(
        [
            _response_for_batch(5),
            _response_for_batch(5),
            _response_for_batch(2),
        ]
    )

    summaries = summarize_clusters(clusters, config, client=client)

    assert len(client.chat.completions.calls) == 3
    assert len(summaries) == 12
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_summarize_clusters_parses_json_to_cluster_summaries(sample_config, make_cluster, make_article) -> None:
    """Map-stage JSON should map back onto cluster metadata."""

    primary = make_article(
        guid="nyt-guid",
        link="https://example.com/nyt",
        source_name="New York Times",
        source_slug="nyt",
        published=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
    )
    duplicate = make_article(
        guid="bbc-guid",
        title="Another angle",
        source_name="BBC News",
        source_slug="bbc",
    )
    cluster = make_cluster(cluster_id="cluster-123", primary=primary, duplicates=[duplicate])
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "items": [
                                {
                                    "topic": "经济金融",
                                    "headline_zh": "油价波动加剧",
                                    "summary_zh": "国际油价因局势变化而震荡。",
                                    "importance": 7,
                                    "entities": ["布伦特原油", "中东"],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )
    client = FakeClient([response])

    summaries = summarize_clusters([cluster], sample_config, client=client)

    assert len(summaries) == 1
    assert summaries[0].cluster_id == "cluster-123"
    assert summaries[0].source_names == ["New York Times", "BBC News"]
    assert summaries[0].primary_link == "https://example.com/nyt"
    assert summaries[0].topic == "经济金融"


def test_summarize_clusters_retries_invalid_json(sample_config, make_cluster, monkeypatch) -> None:
    """Invalid JSON should trigger one retry before succeeding."""

    monkeypatch.setattr("src.summarizer_map.time.sleep", lambda _: None)
    client = FakeClient(
        [
            FakeResponse(choices=[FakeChoice(FakeMessage("not-json"))]),
            _response_for_batch(1),
        ]
    )

    summaries = summarize_clusters([make_cluster()], sample_config, client=client)

    assert len(client.chat.completions.calls) == 2
    assert len(summaries) == 1


def test_summarize_clusters_with_usage_aggregates_batch_tokens(sample_config, make_cluster) -> None:
    """Map-stage helper should accumulate token usage across all batches."""

    clusters = [make_cluster(cluster_id=f"cluster-{index}") for index in range(6)]
    client = FakeClient(
        [
            _response_for_batch(5),
            _response_for_batch(1),
        ]
    )

    result = summarize_clusters_with_usage(clusters, sample_config, client=client)

    assert len(result.summaries) == 6
    assert result.token_usage == {"input_tokens": 222, "output_tokens": 444}
    assert result.model == "deepseek-chat"
    assert result.batches_total == 2
    assert result.batches_failed == 0
    assert result.clusters_skipped == 0


def test_summarize_clusters_with_usage_tracks_skipped_batches(sample_config, make_cluster, monkeypatch) -> None:
    """Map-stage usage helper should expose skipped clusters when a batch fails."""

    monkeypatch.setattr("src.summarizer_map.time.sleep", lambda _: None)
    client = FakeClient(
        [
            FakeResponse(choices=[FakeChoice(FakeMessage("not-json"))]),
            FakeResponse(choices=[FakeChoice(FakeMessage("still-not-json"))]),
        ]
    )

    result = summarize_clusters_with_usage([make_cluster()], sample_config, client=client)

    assert result.summaries == []
    assert result.batches_total == 1
    assert result.batches_failed == 1
    assert result.clusters_skipped == 1


def test_summarize_clusters_with_usage_splits_failed_batch(sample_config, make_cluster, monkeypatch) -> None:
    """A failed multi-cluster batch should be retried as smaller batches instead of being dropped."""

    monkeypatch.setattr("src.summarizer_map.time.sleep", lambda _: None)
    config = replace(
        sample_config,
        summarizer=replace(
            sample_config.summarizer,
            map=replace(sample_config.summarizer.map, batch_size=2, max_retries=1),
        ),
    )
    clusters = [make_cluster(cluster_id="cluster-1"), make_cluster(cluster_id="cluster-2")]
    client = FakeClient(
        [
            FakeResponse(
                choices=[
                    FakeChoice(
                        FakeMessage(
                            json.dumps(
                                {
                                    "items": [
                                        {
                                            "topic": "国际政治",
                                            "headline_zh": "只有一条",
                                            "summary_zh": "触发拆分重试。",
                                            "importance": 8,
                                            "entities": ["伊朗"],
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        )
                    )
                ]
            ),
            _response_for_batch(1),
            _response_for_batch(1),
        ]
    )

    result = summarize_clusters_with_usage(clusters, config, client=client)

    assert len(result.summaries) == 2
    assert result.batches_failed == 1
    assert result.clusters_skipped == 0
