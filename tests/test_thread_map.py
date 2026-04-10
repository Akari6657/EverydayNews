"""Tests for map-stage summarization over story threads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.models import StoryThread
from src.summarizer_map import summarize_threads_with_usage


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

    prompt_tokens: int = 50
    completion_tokens: int = 70


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


def test_summarize_threads_with_usage_returns_cluster_summaries(sample_config, make_article) -> None:
    """Thread map-stage should return compatibility summaries for downstream reduce."""

    article = make_article(
        guid="thread-a1",
        title="Oil prices plunge after ceasefire deal",
        description="Markets react to the latest truce.",
    )
    thread = StoryThread(
        thread_id=7,
        topic="停火影响",
        topic_en="Ceasefire impact",
        articles=[article],
        source_names=[article.source_name, "BBC News"],
        source_count=2,
        primary=article,
        latest_published=article.published,
        rationale="测试",
    )
    response = FakeResponse(
        choices=[
            FakeChoice(
                FakeMessage(
                    json.dumps(
                        {
                            "items": [
                                {
                                    "topic": "经济金融",
                                    "headline_zh": "停火后油价回落",
                                    "summary_zh": "停火消息带动油价与股市反应。",
                                    "importance": 8,
                                    "entities": ["油价", "股市"],
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

    result = summarize_threads_with_usage([thread], sample_config, client=client)

    assert len(result.summaries) == 1
    assert result.summaries[0].cluster_id == "thread-7"
    assert result.summaries[0].topic == "经济金融"
    assert result.summaries[0].source_names == ["New York Times", "BBC News"]
    assert result.token_usage == {"input_tokens": 50, "output_tokens": 70}
