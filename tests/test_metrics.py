"""Tests for run-metrics persistence helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.metrics import RunMetrics, save_run_metrics, subtract_token_usage


def test_save_run_metrics_appends_jsonl(tmp_path) -> None:
    """Saving run metrics should append one JSON object per line."""

    first = RunMetrics(
        generated_at=datetime(2026, 4, 16, 8, 0, tzinfo=timezone.utc),
        mode="run",
        success=True,
        config_path="config.yaml",
        fetched_articles=12,
        map_token_usage={"input_tokens": 100, "output_tokens": 20},
    )
    second = RunMetrics(
        generated_at=datetime(2026, 4, 16, 9, 0, tzinfo=timezone.utc),
        mode="dry_run",
        success=True,
        config_path="config.yaml",
        fetched_articles=5,
    )

    output_path = save_run_metrics(first, tmp_path)
    save_run_metrics(second, tmp_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    first_payload = json.loads(lines[0])
    second_payload = json.loads(lines[1])

    assert output_path.exists()
    assert output_path.name == "metrics.jsonl"
    assert first_payload["date"] == "2026-04-16"
    assert first_payload["mode"] == "run"
    assert first_payload["map_token_usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert second_payload["mode"] == "dry_run"
    assert second_payload["fetched_articles"] == 5


def test_subtract_token_usage_clamps_missing_and_negative_values() -> None:
    """Token subtraction should handle missing keys and clamp at zero."""

    usage = subtract_token_usage(
        {"input_tokens": 5},
        {"input_tokens": 10, "output_tokens": 3},
    )

    assert usage == {"input_tokens": 0, "output_tokens": 0}
