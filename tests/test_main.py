"""Tests for the main orchestration entry points."""

from __future__ import annotations

from pathlib import Path

from src import main


def test_run_pipeline_dry_run_skips_llm(monkeypatch, sample_config, make_article) -> None:
    """Dry-run mode should stop before summarization."""

    monkeypatch.setattr(main, "get_config", lambda _: sample_config)
    monkeypatch.setattr(main, "fetch_all_feeds", lambda config: [make_article()])
    monkeypatch.setattr(main, "deduplicate_articles", lambda articles, config: articles)
    monkeypatch.setattr(main, "summarize_articles", lambda *_: (_ for _ in ()).throw(AssertionError("should not call LLM")))

    results = main.run_pipeline("config.yaml", dry_run=True)

    assert len(results) == 1


def test_parse_run_at_rejects_invalid_times() -> None:
    """Scheduler time parser should reject invalid values."""

    try:
        main._parse_run_at("25:99")
    except ValueError:
        assert True
        return
    raise AssertionError("Expected ValueError for invalid schedule")
