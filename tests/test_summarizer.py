"""Tests for DeepSeek summarization batching and retries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.summarizer import summarize_articles


@dataclass
class FakeUsage:
    """Fake token usage payload."""

    prompt_tokens: int = 321
    completion_tokens: int = 123


@dataclass
class FakeMessage:
    """Fake assistant message."""

    content: str


@dataclass
class FakeChoice:
    """Fake choice wrapper."""

    message: FakeMessage


@dataclass
class FakeResponse:
    """Fake OpenAI-compatible response."""

    choices: list[FakeChoice]
    usage: FakeUsage
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


def test_summarize_articles_batches_all_articles(sample_config, make_article) -> None:
    """Summarizer should send all articles in one request."""

    client = FakeClient(
        [
            FakeResponse(
                choices=[FakeChoice(FakeMessage("## 国际政治\n- 摘要内容"))],
                usage=FakeUsage(),
            )
        ]
    )

    briefing = summarize_articles(
        [make_article(guid="a"), make_article(guid="b", title="Another title")],
        sample_config,
        client=client,
        now=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
    )

    prompt = client.chat.completions.calls[0]["messages"][1]["content"]
    assert "标题: Markets rally as inflation cools" in prompt
    assert "标题: Another title" in prompt
    assert briefing.token_usage["input_tokens"] == 321
    assert briefing.content.startswith("## 国际政治")


def test_summarize_articles_retries_after_transient_failure(sample_config, make_article, monkeypatch) -> None:
    """Summarizer should retry temporary API failures."""

    monkeypatch.setattr("src.summarizer.time.sleep", lambda _: None)
    client = FakeClient(
        [
            RuntimeError("rate limited"),
            FakeResponse(
                choices=[FakeChoice(FakeMessage("## 科技\n- 重试成功"))],
                usage=FakeUsage(),
            ),
        ]
    )

    briefing = summarize_articles([make_article()], sample_config, client=client)

    assert len(client.chat.completions.calls) == 2
    assert "重试成功" in briefing.content
