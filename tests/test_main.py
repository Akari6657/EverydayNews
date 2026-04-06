"""Tests for the main orchestration entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src import main
from src.models import ClusterSummary, FinalBriefing, MapSummariesResult


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


def test_parse_run_at_rejects_invalid_times() -> None:
    """Scheduler time parser should reject invalid values."""

    try:
        main._parse_run_at("25:99")
    except ValueError:
        assert True
        return
    raise AssertionError("Expected ValueError for invalid schedule")
