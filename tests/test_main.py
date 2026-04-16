"""Tests for the main orchestration entry points."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import main
from src import evaluator as evaluator_module
from src.models import (
    EvalResult,
    FinalBriefing,
    MapSummariesResult,
    StoryThread,
    ThreadDedupDiagnostics,
    ThreadSummary,
)


def test_run_pipeline_dry_run_uses_thread_pipeline(monkeypatch, sample_config, make_article) -> None:
    """Default dry-run should stop after the V2 story-thread pipeline pre-LLM steps."""

    article = make_article()
    thread = StoryThread(
        thread_id=1,
        topic="测试故事线",
        topic_en="Test story thread",
        articles=[article],
        source_names=[article.source_name],
        source_count=1,
        primary=article,
        latest_published=article.published,
        rationale="调试用",
    )
    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, config: [thread])
    monkeypatch.setattr(main, "rank_threads", lambda threads, config: threads)
    monkeypatch.setattr(
        main,
        "summarize_threads_with_usage",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not call map-stage LLM")),
    )

    results = main.run_pipeline("config.yaml", dry_run=True)

    assert len(results) == 1
    assert results[0].topic == "测试故事线"


def test_run_pipeline_dump_threads_uses_thread_clusterer(monkeypatch, sample_config, make_article) -> None:
    """Thread dump mode should stop after story-thread clustering."""

    article = make_article()
    thread = StoryThread(
        thread_id=1,
        topic="测试故事线",
        topic_en="Test story thread",
        articles=[article],
        source_names=[article.source_name],
        source_count=1,
        primary=article,
        latest_published=article.published,
        rationale="调试用",
    )
    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, config: [thread])
    monkeypatch.setattr(
        main,
        "summarize_threads_with_usage",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not call map-stage LLM")),
    )

    results = main.run_pipeline("config.yaml", dump_threads=True)

    assert len(results) == 1
    assert results[0].topic == "测试故事线"


def test_run_pipeline_dump_threads_can_apply_within_thread_dedup(
    monkeypatch,
    sample_config,
    make_article,
) -> None:
    """Thread dump mode should support the experimental within-thread dedup pass."""

    article = make_article()
    thread = StoryThread(
        thread_id=1,
        topic="测试故事线",
        topic_en="Test story thread",
        articles=[article],
        source_names=[article.source_name],
        source_count=1,
        primary=article,
        latest_published=article.published,
        rationale="调试用",
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, config: [thread])

    def fake_within_thread_dedup(thread_arg, config_arg):
        calls["dedup_called"] = True
        return thread_arg, ThreadDedupDiagnostics(before_articles=1, after_articles=1)

    monkeypatch.setattr(main, "deduplicate_within_thread_with_diagnostics", fake_within_thread_dedup)

    results = main.run_pipeline(
        "config.yaml",
        dump_threads=True,
        dedup_within_threads=True,
    )

    assert len(results) == 1
    assert calls["dedup_called"] is True


def test_run_pipeline_uses_story_threads_by_default(
    monkeypatch,
    sample_config,
    make_article,
    tmp_path: Path,
) -> None:
    """The default pipeline should use story threads end to end."""

    article = make_article(guid="thread-article-1")
    thread = StoryThread(
        thread_id=1,
        topic="测试故事线",
        topic_en="Test story thread",
        articles=[article],
        source_names=[article.source_name],
        source_count=1,
        primary=article,
        latest_published=article.published,
        rationale="调试用",
    )
    summary = ThreadSummary(
        thread_id="thread-1",
        topic="国际政治",
        headline_zh="线程摘要标题",
        summary_zh="线程摘要内容",
        importance=8,
        entities=["实体A"],
        source_names=[article.source_name],
        primary_link=article.link,
    )
    briefing = FinalBriefing(
        date="2026-04-08",
        overview_zh="线程版综述。",
        top_stories=[summary],
        other_stories=[],
        total_threads=1,
        total_sources=1,
        total_articles=1,
        generated_at=datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 10, "output_tokens": 20},
        model="deepseek-chat",
    )
    output_path = tmp_path / "briefing-2026-04-08.md"
    calls: dict[str, object] = {}

    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, config: [thread])
    monkeypatch.setattr(main, "rank_threads", lambda threads, config: threads)
    monkeypatch.setattr(
        main,
        "summarize_threads_with_usage",
        lambda threads, config: MapSummariesResult(
            summaries=[summary],
            token_usage={"input_tokens": 10, "output_tokens": 20},
            model="deepseek-chat",
        ),
    )

    def fake_build_final_briefing(summaries, config, token_usage):
        calls["summaries"] = summaries
        return briefing

    monkeypatch.setattr(main, "build_final_briefing", fake_build_final_briefing)
    monkeypatch.setattr(main, "format_briefing", lambda briefing_arg, config_arg: output_path)
    monkeypatch.setattr(main, "notify", lambda *args: None)

    result = main.run_pipeline("config.yaml")

    assert result == output_path
    assert calls["summaries"] == [summary]


def test_run_pipeline_writes_metrics_jsonl(
    monkeypatch,
    sample_config,
    make_article,
    tmp_path: Path,
) -> None:
    """Successful full runs should append a metrics record under output/metrics.jsonl."""

    article = make_article(guid="thread-article-1")
    thread = StoryThread(
        thread_id=1,
        topic="测试故事线",
        topic_en="Test story thread",
        articles=[article],
        source_names=[article.source_name],
        source_count=1,
        primary=article,
        latest_published=article.published,
        rationale="调试用",
    )
    summary = ThreadSummary(
        thread_id=1,
        topic="国际政治",
        headline_zh="线程摘要标题",
        summary_zh="线程摘要内容",
        importance=8,
        entities=["实体A"],
        source_names=[article.source_name],
        primary_link=article.link,
        source_count=1,
        article_count=1,
    )
    briefing = FinalBriefing(
        date="2026-04-08",
        overview_zh="线程版综述。",
        top_stories=[summary],
        other_stories=[],
        total_threads=1,
        total_sources=1,
        total_articles=1,
        generated_at=datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 15, "output_tokens": 25},
        model="deepseek-chat",
    )
    output_path = tmp_path / "briefing-2026-04-08.md"

    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, config: [thread])
    monkeypatch.setattr(main, "rank_threads", lambda threads, config: threads)
    monkeypatch.setattr(
        main,
        "summarize_threads_with_usage",
        lambda threads, config: MapSummariesResult(
            summaries=[summary],
            token_usage={"input_tokens": 10, "output_tokens": 20},
            model="deepseek-chat",
            batches_total=1,
            batches_failed=0,
            threads_skipped=0,
        ),
    )
    monkeypatch.setattr(main, "build_final_briefing", lambda summaries, config, token_usage: briefing)
    monkeypatch.setattr(main, "format_briefing", lambda briefing_arg, config_arg: output_path)
    monkeypatch.setattr(main, "notify", lambda *args: None)

    result = main.run_pipeline("config.yaml")

    metrics_path = sample_config.root_dir / "output" / "metrics.jsonl"
    payload = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])

    assert result == output_path
    assert payload["success"] is True
    assert payload["mode"] == "run"
    assert payload["fetched_articles"] == 1
    assert payload["threads_after_clustering"] == 1
    assert payload["threads_after_within_thread_dedup"] == 1
    assert payload["threads_after_ranking"] == 1
    assert payload["map_summaries"] == 1
    assert payload["reduce_candidates"] == 1
    assert payload["output_path"] == str(output_path)
    assert payload["map_token_usage"] == {"input_tokens": 10, "output_tokens": 20}
    assert payload["reduce_token_usage"] == {"input_tokens": 5, "output_tokens": 5}
    assert payload["total_token_usage"] == {"input_tokens": 15, "output_tokens": 25}
    assert payload["reduce_fallback"] is False


def test_run_pipeline_records_eval_metrics(
    monkeypatch,
    sample_config,
    make_article,
    tmp_path: Path,
) -> None:
    """Eval-enabled runs should record eval outputs and token usage in metrics."""

    article = make_article(guid="thread-article-1")
    thread = StoryThread(
        thread_id=1,
        topic="测试故事线",
        topic_en="Test story thread",
        articles=[article],
        source_names=[article.source_name],
        source_count=1,
        primary=article,
        latest_published=article.published,
        rationale="调试用",
    )
    summary = ThreadSummary(
        thread_id=1,
        topic="国际政治",
        headline_zh="线程摘要标题",
        summary_zh="线程摘要内容",
        importance=8,
        entities=[],
        source_names=[article.source_name],
        primary_link=article.link,
        source_count=1,
        article_count=1,
    )
    briefing = FinalBriefing(
        date="2026-04-08",
        overview_zh="线程版综述。",
        top_stories=[summary],
        other_stories=[],
        total_threads=1,
        total_sources=1,
        total_articles=1,
        generated_at=datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 10, "output_tokens": 20},
        model="deepseek-chat",
    )
    eval_result = EvalResult(
        date="2026-04-08",
        coverage=8,
        diversity=7,
        clarity=9,
        redundancy=8,
        importance_calibration=7,
        notes="整体质量良好",
        model="deepseek-chat",
        generated_at=datetime(2026, 4, 8, 12, 5, tzinfo=timezone.utc),
        token_usage={"input_tokens": 30, "output_tokens": 8},
        candidate_count=1,
        briefing_thread_count=1,
    )
    output_path = tmp_path / "briefing-2026-04-08.md"
    eval_path = tmp_path / "output" / "eval" / "eval-2026-04-08.json"

    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, config: [thread])
    monkeypatch.setattr(main, "rank_threads", lambda threads, config: threads)
    monkeypatch.setattr(
        main,
        "summarize_threads_with_usage",
        lambda threads, config: MapSummariesResult(
            summaries=[summary],
            token_usage={"input_tokens": 10, "output_tokens": 20},
            model="deepseek-chat",
            batches_total=1,
            batches_failed=0,
            threads_skipped=0,
        ),
    )
    monkeypatch.setattr(main, "build_final_briefing", lambda summaries, config, token_usage: briefing)
    monkeypatch.setattr(main, "format_briefing", lambda briefing_arg, config_arg: output_path)
    monkeypatch.setattr(main, "notify", lambda *args: None)
    monkeypatch.setattr(evaluator_module, "evaluate_briefing", lambda *args: eval_result)
    monkeypatch.setattr(evaluator_module, "save_eval_result", lambda *args: eval_path)

    result = main.run_pipeline("config.yaml", eval=True)

    metrics_path = sample_config.root_dir / "output" / "metrics.jsonl"
    payload = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])

    assert result == output_path
    assert payload["eval_enabled"] is True
    assert payload["eval_output_path"] == str(eval_path)
    assert payload["eval_token_usage"] == {"input_tokens": 30, "output_tokens": 8}


def test_run_pipeline_records_failure_metrics(
    monkeypatch,
    sample_config,
    make_article,
) -> None:
    """Failed runs should still persist error details in run metrics."""

    article = make_article(guid="thread-article-1")
    thread = StoryThread(
        thread_id=1,
        topic="测试故事线",
        topic_en="Test story thread",
        articles=[article],
        source_names=[article.source_name],
        source_count=1,
        primary=article,
        latest_published=article.published,
        rationale="调试用",
    )

    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, config: [thread])
    monkeypatch.setattr(main, "rank_threads", lambda threads, config: threads)
    monkeypatch.setattr(
        main,
        "summarize_threads_with_usage",
        lambda *_: (_ for _ in ()).throw(RuntimeError("map stage failed")),
    )

    with pytest.raises(RuntimeError, match="map stage failed"):
        main.run_pipeline("config.yaml")

    metrics_path = sample_config.root_dir / "output" / "metrics.jsonl"
    payload = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])

    assert payload["success"] is False
    assert payload["mode"] == "run"
    assert payload["fetched_articles"] == 1
    assert payload["threads_after_ranking"] == 1
    assert payload["map_summaries"] is None
    assert payload["error"] == "RuntimeError: map stage failed"


def test_parse_run_at_rejects_invalid_times() -> None:
    """Scheduler time parser should reject invalid values."""

    try:
        main._parse_run_at("25:99")
    except ValueError:
        assert True
        return
    raise AssertionError("Expected ValueError for invalid schedule")
