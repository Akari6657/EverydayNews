"""Tests for LLM-based story-thread clustering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from src.thread_clusterer import cluster_into_threads


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

    prompt_tokens: int = 123
    completion_tokens: int = 45


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


def test_basic_clustering(sample_config, make_article) -> None:
    """Related headlines should be grouped into one story thread."""

    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    articles = [
        make_article(
            guid="a1",
            source_name="New York Times",
            source_slug="nyt",
            category="business",
            title="Oil prices plunge after ceasefire deal",
            published=now,
        ),
        make_article(
            guid="a2",
            source_name="BBC News",
            source_slug="bbc",
            category="world",
            title="Stocks rise as U.S. and Iran agree to ceasefire",
            published=now - timedelta(minutes=5),
        ),
        make_article(
            guid="a3",
            title="Taiwan opposition leader visits China",
            published=now - timedelta(minutes=10),
        ),
        make_article(
            guid="a4",
            source_name="BBC News",
            source_slug="bbc",
            category="business",
            title="Markets cheer as oil tumbles on truce plan",
            published=now - timedelta(minutes=15),
        ),
        make_article(
            guid="a5",
            title="Greece to ban social media for under-15s",
            published=now - timedelta(minutes=20),
        ),
    ]
    client = FakeClient(
        [
            _thread_response(
                [
                    {
                        "thread_id": 1,
                        "topic": "美伊停火",
                        "topic_en": "US-Iran ceasefire",
                        "article_ids": [1, 2, 4],
                        "rationale": "都在讲停火对市场和局势的影响",
                    },
                    {
                        "thread_id": 2,
                        "topic": "两岸关系",
                        "topic_en": "Cross-strait relations",
                        "article_ids": [3],
                        "rationale": "台湾反对党访华",
                    },
                    {
                        "thread_id": 3,
                        "topic": "欧洲政策",
                        "topic_en": "European policy",
                        "article_ids": [5],
                        "rationale": "希腊拟出台社媒禁令",
                    },
                ]
            )
        ]
    )

    threads = cluster_into_threads(articles, sample_config, client=client)

    assert len(threads) == 3
    assert threads[0].topic == "美伊停火"
    assert threads[0].source_count == 2
    assert len(threads[0].articles) == 3
    assert threads[0].primary.source_slug == "nyt"


def test_orphaned_articles_become_singletons(sample_config, make_article, caplog) -> None:
    """Missing article IDs should be kept as singleton fallback threads."""

    articles = [
        make_article(guid="a1", title="Story A"),
        make_article(guid="a2", title="Story B"),
        make_article(guid="a3", title="Story C"),
    ]
    client = FakeClient(
        [
            _thread_response(
                [
                    {
                        "thread_id": 1,
                        "topic": "主题A",
                        "topic_en": "Topic A",
                        "article_ids": [1, 2],
                        "rationale": "前两条被分到同组",
                    }
                ]
            )
        ]
    )

    threads = cluster_into_threads(articles, sample_config, client=client)

    assert any(
        len(thread.articles) == 1
        and thread.articles[0].guid == "a3"
        and thread.rationale == "模型遗漏，已自动单独保留"
        for thread in threads
    )
    assert "orphaned article" in caplog.text


def test_duplicate_assignments_keep_first_thread(sample_config, make_article, caplog) -> None:
    """Duplicate article assignments should only keep the first occurrence."""

    articles = [
        make_article(guid="a1", title="Story A"),
        make_article(guid="a2", title="Story B"),
        make_article(guid="a3", title="Story C"),
    ]
    client = FakeClient(
        [
            _thread_response(
                [
                    {
                        "thread_id": 1,
                        "topic": "主题A",
                        "topic_en": "Topic A",
                        "article_ids": [1, 2],
                        "rationale": "第一组",
                    },
                    {
                        "thread_id": 2,
                        "topic": "主题B",
                        "topic_en": "Topic B",
                        "article_ids": [2, 3],
                        "rationale": "第二组",
                    },
                ]
            )
        ]
    )

    threads = cluster_into_threads(articles, sample_config, client=client)

    assert sum(article.guid == "a2" for thread in threads for article in thread.articles) == 1
    assert "assigned to multiple threads" in caplog.text


def test_invalid_json_retry(sample_config, make_article, monkeypatch) -> None:
    """Invalid JSON should trigger one retry before succeeding."""

    monkeypatch.setattr("src.thread_clusterer.time.sleep", lambda _: None)
    articles = [make_article(guid="a1", title="Story A")]
    client = FakeClient(
        [
            FakeResponse(choices=[FakeChoice(FakeMessage("not-json"))]),
            _thread_response(
                [
                    {
                        "thread_id": 1,
                        "topic": "主题A",
                        "topic_en": "Topic A",
                        "article_ids": [1],
                        "rationale": "有效重试",
                    }
                ]
            ),
        ]
    )

    threads = cluster_into_threads(articles, sample_config, client=client)

    assert len(threads) == 1
    assert len(client.chat.completions.calls) == 2


def test_invalid_json_falls_back_to_one_per_thread(sample_config, make_article, monkeypatch) -> None:
    """Two invalid JSON responses should degrade to one article per thread."""

    monkeypatch.setattr("src.thread_clusterer.time.sleep", lambda _: None)
    articles = [
        make_article(guid="a1", title="Story A"),
        make_article(guid="a2", title="Story B"),
    ]
    client = FakeClient(
        [
            FakeResponse(choices=[FakeChoice(FakeMessage("not-json"))]),
            FakeResponse(choices=[FakeChoice(FakeMessage("still-not-json"))]),
        ]
    )

    threads = cluster_into_threads(articles, sample_config, client=client)

    assert len(threads) == 2
    assert all(len(thread.articles) == 1 for thread in threads)


def test_empty_input_skips_llm(sample_config) -> None:
    """Empty input should return immediately without any client calls."""

    client = FakeClient([])

    threads = cluster_into_threads([], sample_config, client=client)

    assert threads == []
    assert client.chat.completions.calls == []


def test_generic_wrapper_titles_become_singleton_threads(sample_config, make_article) -> None:
    """Generic wrapper headlines should be isolated before clustering."""

    articles = [
        make_article(guid="a1", title="Morning news brief"),
        make_article(guid="a2", title="Taiwan opposition leader visits China"),
    ]
    client = FakeClient(
        [
            _thread_response(
                [
                    {
                        "thread_id": 1,
                        "topic": "两岸关系",
                        "topic_en": "Cross-strait relations",
                        "article_ids": [1],
                        "rationale": "访华相关新闻",
                    }
                ]
            )
        ]
    )

    threads = cluster_into_threads(articles, sample_config, client=client)

    assert len(threads) == 2
    assert any(thread.articles[0].title == "Morning news brief" for thread in threads)


def test_oversized_thread_is_refined(sample_config, make_article, monkeypatch) -> None:
    """Oversized threads should be re-split into tighter subthreads."""

    monkeypatch.setattr("src.thread_clusterer.time.sleep", lambda _: None)
    config = replace(
        sample_config,
        thread_clustering=replace(
            sample_config.thread_clustering,
            max_articles_per_thread=3,
            max_refinement_rounds=1,
        ),
    )
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    articles = [
        make_article(guid="a1", title="Ceasefire agreed", published=now),
        make_article(guid="a2", title="Ceasefire terms draw scrutiny", published=now - timedelta(minutes=1)),
        make_article(guid="a3", title="Oil prices plunge after the truce", published=now - timedelta(minutes=2)),
        make_article(guid="a4", title="Shipping remains throttled in Hormuz", published=now - timedelta(minutes=3)),
        make_article(guid="a5", title="Taiwan opposition leader visits China", published=now - timedelta(minutes=4)),
    ]
    client = FakeClient(
        [
            _thread_response(
                [
                    {
                        "thread_id": 1,
                        "topic": "美伊停火",
                        "topic_en": "US-Iran ceasefire",
                        "article_ids": [1, 2, 3, 4],
                        "rationale": "初次分组过粗",
                    },
                    {
                        "thread_id": 2,
                        "topic": "两岸关系",
                        "topic_en": "Cross-strait relations",
                        "article_ids": [5],
                        "rationale": "访华相关新闻",
                    },
                ]
            ),
            _thread_response(
                [
                    {
                        "thread_id": 1,
                        "topic": "停火谈判",
                        "topic_en": "Ceasefire diplomacy",
                        "article_ids": [1, 2],
                        "rationale": "停火条款与解释",
                    },
                    {
                        "thread_id": 2,
                        "topic": "市场影响",
                        "topic_en": "Market impact",
                        "article_ids": [3, 4],
                        "rationale": "经济与航运后果",
                    },
                ]
            ),
        ]
    )

    threads = cluster_into_threads(articles, config, client=client)

    assert len(threads) == 3
    assert {thread.topic for thread in threads} >= {"停火谈判", "市场影响", "两岸关系"}


def _thread_response(threads: list[dict[str, object]]) -> FakeResponse:
    """Return one valid thread-clustering response."""

    payload = {"threads": threads}
    return FakeResponse(choices=[FakeChoice(FakeMessage(json.dumps(payload, ensure_ascii=False)))])
