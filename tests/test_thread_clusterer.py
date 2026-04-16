"""Tests for LLM-based story-thread clustering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from src.thread_clusterer import (
    _is_generic_wrapper_title,
    _merge_chunk_threads_via_llm,
    _merge_overlapping_threads,
    cluster_into_threads,
)


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


def _merge_response(merges: list[dict[str, object]]) -> FakeResponse:
    """Return one valid thread-merge response."""

    payload = {"merges": merges}
    return FakeResponse(choices=[FakeChoice(FakeMessage(json.dumps(payload, ensure_ascii=False)))])


# ---------------------------------------------------------------------------
# Expanded wrapper-title detection
# ---------------------------------------------------------------------------


def test_expanded_wrapper_title_detection(sample_config, make_article):
    """New wrapper patterns are isolated as singletons without LLM clustering."""

    wrapper_titles = [
        "Up First: Trump signs Iran deal",
        "Up First Newsletter — what you need to know",
        "Daily Briefing: top stories today",
        "Evening Briefing: the news at dusk",
        "Week in Review: biggest stories",
    ]
    articles = [make_article(title=t, source_name="NPR") for t in wrapper_titles]
    # One regular article so LLM would normally be called
    articles.append(make_article(title="Iran ceasefire signed", source_name="BBC News"))

    client = FakeClient([
        _thread_response([{"thread_id": 1, "topic": "伊朗停火", "topic_en": "Iran Ceasefire", "article_ids": [1], "rationale": "only real article"}])
    ])
    threads = cluster_into_threads(articles, sample_config, client=client)

    wrapper_thread_topics = {t.topic for t in threads if len(t.articles) == 1 and _is_generic_wrapper_title(t.articles[0].title)}
    assert len(wrapper_thread_topics) == len(wrapper_titles)
    # LLM was called exactly once (for the one real article)
    assert len(client.chat.completions.calls) == 1


# ---------------------------------------------------------------------------
# Heuristic cross-thread merge
# ---------------------------------------------------------------------------


def test_overlapping_threads_merged_by_heuristic(sample_config, make_article):
    """Two threads with highly overlapping article titles are merged heuristically."""

    config = replace(
        sample_config,
        thread_clustering=replace(
            sample_config.thread_clustering,
            enable_post_merge=True,
            merge_overlap_threshold=0.20,
        ),
    )
    # Both threads are about Iran ceasefire from different angles; many shared tokens
    articles_a = [
        make_article(title="Iran ceasefire talks progress in Geneva", source_name="NYT", link="https://nyt.com/1"),
        make_article(title="Iran nuclear ceasefire deal signed", source_name="BBC News", link="https://bbc.com/1"),
    ]
    articles_b = [
        make_article(title="Iran ceasefire agreement reached after talks", source_name="Guardian", link="https://guardian.com/1"),
    ]

    client = FakeClient([
        _thread_response([
            {"thread_id": 1, "topic": "伊朗停火谈判", "topic_en": "Iran Ceasefire Talks", "article_ids": [1, 2], "rationale": "same ceasefire"},
            {"thread_id": 2, "topic": "伊朗停火协议", "topic_en": "Iran Ceasefire Agreement", "article_ids": [3], "rationale": "agreement signed"},
        ])
    ])
    threads = cluster_into_threads(articles_a + articles_b, config, client=client)

    # The two Iran ceasefire threads should have been merged into one
    assert len(threads) == 1
    assert len(threads[0].articles) == 3


def test_distinct_threads_not_merged(sample_config, make_article):
    """Threads about unrelated events are not merged by the heuristic."""

    config = replace(
        sample_config,
        thread_clustering=replace(
            sample_config.thread_clustering,
            enable_post_merge=True,
            merge_overlap_threshold=0.30,
        ),
    )
    articles = [
        make_article(title="Iran ceasefire signed in Geneva", source_name="NYT"),
        make_article(title="European carbon tax reform passed", source_name="BBC News"),
    ]
    client = FakeClient([
        _thread_response([
            {"thread_id": 1, "topic": "伊朗停火", "topic_en": "Iran Ceasefire", "article_ids": [1], "rationale": "ceasefire"},
            {"thread_id": 2, "topic": "欧盟碳税", "topic_en": "EU Carbon Tax", "article_ids": [2], "rationale": "carbon tax"},
        ])
    ])
    threads = cluster_into_threads(articles, config, client=client)

    assert len(threads) == 2


# ---------------------------------------------------------------------------
# LLM merge pass for chunked clustering
# ---------------------------------------------------------------------------


def test_chunk_merge_llm_pass(sample_config, make_article, monkeypatch):
    """After chunked clustering, the LLM merge pass combines duplicate threads."""

    monkeypatch.setattr("src.thread_clusterer.time.sleep", lambda _: None)
    config = replace(
        sample_config,
        thread_clustering=replace(
            sample_config.thread_clustering,
            max_articles_per_call=2,   # force chunking with 4 articles
            enable_chunk_merge=True,
            enable_post_merge=False,
        ),
    )
    # 4 articles split into 2 chunks of 2; each chunk produces one Iran thread
    articles = [
        make_article(title="Iran ceasefire talks", source_name="NYT", link="https://nyt.com/1"),
        make_article(title="Iran diplomacy progress", source_name="BBC News", link="https://bbc.com/1"),
        make_article(title="Iran deal signed today", source_name="Guardian", link="https://guardian.com/1"),
        make_article(title="Iran nuclear agreement reached", source_name="Al Jazeera English", link="https://aje.com/1"),
    ]

    # Chunk 1 produces thread_id=1 (Iran), chunk 2 produces thread_id=1 (also Iran)
    # Then the merge pass sees two threads and merges them
    client = FakeClient([
        # Chunk 1 clustering
        _thread_response([{"thread_id": 1, "topic": "伊朗谈判", "topic_en": "Iran Talks", "article_ids": [1, 2], "rationale": "chunk1"}]),
        # Chunk 2 clustering
        _thread_response([{"thread_id": 1, "topic": "伊朗协议", "topic_en": "Iran Deal", "article_ids": [1, 2], "rationale": "chunk2"}]),
        # LLM merge pass
        _merge_response([{"ids": [1, 2], "topic": "伊朗停火与外交", "topic_en": "Iran Ceasefire and Diplomacy"}]),
    ])
    threads = cluster_into_threads(articles, config, client=client)

    assert len(threads) == 1
    assert threads[0].topic == "伊朗停火与外交"
    assert len(threads[0].articles) == 4


def test_chunk_merge_falls_back_on_llm_failure(sample_config, make_article, monkeypatch):
    """If the merge LLM call raises an exception, threads are returned unchanged."""

    monkeypatch.setattr("src.thread_clusterer.time.sleep", lambda _: None)
    config = replace(
        sample_config,
        thread_clustering=replace(
            sample_config.thread_clustering,
            max_articles_per_call=2,
            max_retries=1,
            enable_chunk_merge=True,
            enable_post_merge=False,
        ),
    )
    articles = [
        make_article(title="Iran ceasefire talks", source_name="NYT"),
        make_article(title="Iran diplomacy progress", source_name="BBC News"),
        make_article(title="EU carbon tax vote", source_name="Guardian"),
        make_article(title="European emissions reform", source_name="DW"),
    ]
    client = FakeClient([
        _thread_response([{"thread_id": 1, "topic": "伊朗谈判", "topic_en": "Iran", "article_ids": [1, 2], "rationale": ""}]),
        _thread_response([{"thread_id": 1, "topic": "欧盟碳税", "topic_en": "EU Tax", "article_ids": [1, 2], "rationale": ""}]),
        RuntimeError("LLM unavailable"),   # merge pass fails
    ])
    threads = cluster_into_threads(articles, config, client=client)

    # Fallback: both original threads kept
    assert len(threads) == 2
