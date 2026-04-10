"""Tests for the main orchestration entry points."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from src import main
from src.models import (
    ClusterSummary,
    EvaluationConfig,
    EvaluationResult,
    FinalBriefing,
    MapSummariesResult,
    StoryThread,
    ThreadDedupDiagnostics,
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
    """The default pipeline should use story threads instead of article clusters."""

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
    summary = ClusterSummary(
        cluster_id="thread-1",
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
        topics={"国际政治": [summary]},
        total_clusters=1,
        total_sources=1,
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
    monkeypatch.setattr(main, "format_briefing", lambda briefing_arg, articles_arg, config_arg: output_path)
    monkeypatch.setattr(main, "notify", lambda *args: None)

    result = main.run_pipeline("config.yaml")

    assert result == output_path
    assert calls["summaries"] == [summary]

def test_run_pipeline_writes_evaluation_when_enabled(
    monkeypatch,
    sample_config,
    make_article,
    tmp_path: Path,
) -> None:
    """Full pipeline should invoke evaluation and metrics hooks when enabled."""

    config = replace(
        sample_config,
        evaluation=EvaluationConfig(enabled=True, max_retries=2),
    )
    article = make_article(guid="article-1")
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
    briefing = FinalBriefing(
        date="2026-04-06",
        overview_zh="今日综述。",
        topics={"国际政治": [summary]},
        total_clusters=1,
        total_sources=1,
        generated_at=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 12, "output_tokens": 34},
        model="deepseek-chat",
    )
    output_path = tmp_path / "briefing-2026-04-06.md"
    evaluation_result = EvaluationResult(
        coverage=8,
        diversity=7,
        clarity=9,
        redundancy=8,
        importance_calibration=7,
        notes="稳定。",
        token_usage={"input_tokens": 5, "output_tokens": 6},
        model="deepseek-chat",
        generated_at=datetime(2026, 4, 6, 12, 1, tzinfo=timezone.utc),
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(main, "get_config", lambda _: config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda current_config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, current_config: [thread])
    monkeypatch.setattr(main, "rank_threads", lambda threads, current_config: threads)
    monkeypatch.setattr(
        main,
        "summarize_threads_with_usage",
        lambda threads, current_config: MapSummariesResult(
            summaries=[summary],
            token_usage={"input_tokens": 10, "output_tokens": 20},
            model="deepseek-chat",
            batches_total=1,
            batches_failed=0,
            clusters_skipped=0,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_final_briefing",
        lambda summaries, current_config, token_usage: briefing,
    )

    def fake_format_briefing(briefing_arg, articles_arg, config_arg):
        output_path.write_text("# 简报\n\n测试内容\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(main, "format_briefing", fake_format_briefing)
    monkeypatch.setattr(main, "notify", lambda *args: None)

    def fake_evaluate_briefing(markdown_content, summaries, current_config):
        calls["evaluation_markdown"] = markdown_content
        calls["evaluation_summaries"] = summaries
        return evaluation_result

    def fake_write_evaluation_result(result, generated_at, current_config):
        calls["evaluation_result"] = result
        calls["evaluation_generated_at"] = generated_at
        return tmp_path / "briefing-2026-04-06.eval.json"

    def fake_append_run_metrics(metrics, current_config):
        calls["metrics"] = metrics
        return tmp_path / "metrics.jsonl"

    monkeypatch.setattr(main, "evaluate_briefing", fake_evaluate_briefing)
    monkeypatch.setattr(main, "write_evaluation_result", fake_write_evaluation_result)
    monkeypatch.setattr(main, "append_run_metrics", fake_append_run_metrics)

    result = main.run_pipeline("config.yaml", dry_run=False)

    assert result == output_path
    assert calls["evaluation_markdown"] == "# 简报\n\n测试内容\n"
    assert calls["evaluation_summaries"] == [summary]
    assert calls["evaluation_result"] == evaluation_result
    assert calls["evaluation_generated_at"] == briefing.generated_at
    assert calls["metrics"].eval_scores == evaluation_result.scores
    assert calls["metrics"].status == "success"


def test_run_pipeline_appends_failed_metrics(monkeypatch, sample_config, make_article, tmp_path: Path) -> None:
    """Failures after map-stage should still append a failed metrics record."""

    article = make_article(guid="article-1")
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
    briefing = FinalBriefing(
        date="2026-04-06",
        overview_zh="今日综述。",
        topics={"国际政治": [summary]},
        total_clusters=1,
        total_sources=1,
        generated_at=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 12, "output_tokens": 34},
        model="deepseek-chat",
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda current_config: [article])
    monkeypatch.setattr(main, "cluster_into_threads", lambda articles, current_config: [thread])
    monkeypatch.setattr(main, "rank_threads", lambda threads, current_config: threads)
    monkeypatch.setattr(
        main,
        "summarize_threads_with_usage",
        lambda threads, current_config: MapSummariesResult(
            summaries=[summary],
            token_usage={"input_tokens": 10, "output_tokens": 20},
            model="deepseek-chat",
            batches_total=1,
            batches_failed=0,
            clusters_skipped=0,
        ),
    )
    monkeypatch.setattr(
        main,
        "build_final_briefing",
        lambda summaries, current_config, token_usage: briefing,
    )
    monkeypatch.setattr(
        main,
        "format_briefing",
        lambda *args: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    def fake_append_run_metrics(metrics, current_config):
        calls["metrics"] = metrics
        return tmp_path / "metrics.jsonl"

    monkeypatch.setattr(main, "append_run_metrics", fake_append_run_metrics)

    try:
        main.run_pipeline("config.yaml", dry_run=False)
    except RuntimeError as exc:
        assert str(exc) == "disk full"
    else:
        raise AssertionError("Expected RuntimeError from format_briefing")

    assert calls["metrics"].status == "failed"
    assert calls["metrics"].failure_stage == "format"
    assert "disk full" in calls["metrics"].failure_reason


def test_parse_run_at_rejects_invalid_times() -> None:
    """Scheduler time parser should reject invalid values."""

    try:
        main._parse_run_at("25:99")
    except ValueError:
        assert True
        return
    raise AssertionError("Expected ValueError for invalid schedule")
