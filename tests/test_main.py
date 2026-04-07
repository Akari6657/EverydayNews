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
)


def test_run_pipeline_dry_run_skips_llm(monkeypatch, sample_config, make_article, make_cluster) -> None:
    """Dry-run mode should stop before summarization."""

    cluster = make_cluster(primary=make_article())
    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [make_article()])
    monkeypatch.setattr(main, "deduplicate", lambda articles, config: [cluster])
    monkeypatch.setattr(
        main,
        "summarize_clusters_with_usage",
        lambda *_: (_ for _ in ()).throw(AssertionError("should not call map-stage LLM")),
    )

    results = main.run_pipeline("config.yaml", dry_run=True)

    assert len(results) == 1
    assert results[0].cluster_id == cluster.cluster_id


def test_run_pipeline_uses_map_reduce_flow(monkeypatch, sample_config, make_article, make_cluster, tmp_path: Path) -> None:
    """Full pipeline should run through dedup, map, reduce, formatting, and notify."""

    cluster = make_cluster(primary=make_article(guid="article-1"))
    summary = ClusterSummary(
        cluster_id=cluster.cluster_id,
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
    calls: dict[str, object] = {}

    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [make_article()])
    monkeypatch.setattr(main, "deduplicate", lambda articles, config: [cluster])
    monkeypatch.setattr(
        main,
        "summarize_clusters_with_usage",
        lambda clusters, config: MapSummariesResult(
            summaries=[summary],
            token_usage={"input_tokens": 10, "output_tokens": 20},
            model="deepseek-chat",
        ),
    )

    def fake_build_final_briefing(summaries, config, token_usage):
        calls["summaries"] = summaries
        calls["token_usage"] = token_usage
        return briefing

    def fake_format_briefing(briefing_arg, articles_arg, config_arg):
        calls["format_articles_arg"] = articles_arg
        calls["format_briefing_arg"] = briefing_arg
        return output_path

    def fake_notify(path_arg, briefing_arg, config_arg):
        calls["notify_path"] = path_arg
        calls["notify_briefing"] = briefing_arg

    monkeypatch.setattr(main, "build_final_briefing", fake_build_final_briefing)
    monkeypatch.setattr(main, "format_briefing", fake_format_briefing)
    monkeypatch.setattr(main, "notify", fake_notify)

    result = main.run_pipeline("config.yaml", dry_run=False)

    assert result == output_path
    assert calls["summaries"] == [summary]
    assert calls["token_usage"] == {"input_tokens": 10, "output_tokens": 20}
    assert calls["format_articles_arg"] is None
    assert calls["format_briefing_arg"] == briefing
    assert calls["notify_path"] == output_path
    assert calls["notify_briefing"] == briefing


def test_run_pipeline_writes_evaluation_when_enabled(
    monkeypatch,
    sample_config,
    make_article,
    make_cluster,
    tmp_path: Path,
) -> None:
    """Full pipeline should invoke evaluation and metrics hooks when enabled."""

    config = replace(
        sample_config,
        evaluation=EvaluationConfig(enabled=True, max_retries=2),
    )
    cluster = make_cluster(primary=make_article(guid="article-1"))
    summary = ClusterSummary(
        cluster_id=cluster.cluster_id,
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
    monkeypatch.setattr(main, "fetch_all_feeds", lambda current_config: [make_article()])
    monkeypatch.setattr(main, "deduplicate", lambda articles, current_config: [cluster])
    monkeypatch.setattr(
        main,
        "summarize_clusters_with_usage",
        lambda clusters, current_config: MapSummariesResult(
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


def test_run_pipeline_appends_failed_metrics(monkeypatch, sample_config, make_article, make_cluster, tmp_path: Path) -> None:
    """Failures after map-stage should still append a failed metrics record."""

    cluster = make_cluster(primary=make_article(guid="article-1"))
    summary = ClusterSummary(
        cluster_id=cluster.cluster_id,
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
    monkeypatch.setattr(main, "fetch_all_feeds", lambda current_config: [make_article()])
    monkeypatch.setattr(main, "deduplicate", lambda articles, current_config: [cluster])
    monkeypatch.setattr(
        main,
        "summarize_clusters_with_usage",
        lambda clusters, current_config: MapSummariesResult(
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
