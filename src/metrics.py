"""Persist per-run pipeline metrics as JSON Lines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _empty_token_usage() -> dict[str, int]:
    """Return a zero-valued token usage mapping."""

    return {"input_tokens": 0, "output_tokens": 0}


@dataclass(frozen=True)
class RunMetrics:
    """Structured metrics captured for one pipeline invocation."""

    generated_at: datetime
    mode: str
    success: bool
    config_path: str
    eval_enabled: bool = False
    fetched_articles: int = 0
    threads_after_clustering: int | None = None
    threads_after_within_thread_dedup: int | None = None
    within_thread_changed_threads: int = 0
    within_thread_removed_articles: int = 0
    threads_after_ranking: int | None = None
    map_summaries: int | None = None
    reduce_candidates: int | None = None
    briefing_threads: int | None = None
    briefing_sources: int | None = None
    briefing_articles: int | None = None
    output_path: str | None = None
    eval_output_path: str | None = None
    map_token_usage: dict[str, int] = field(default_factory=_empty_token_usage)
    reduce_token_usage: dict[str, int] = field(default_factory=_empty_token_usage)
    total_token_usage: dict[str, int] = field(default_factory=_empty_token_usage)
    eval_token_usage: dict[str, int] = field(default_factory=_empty_token_usage)
    map_batches_total: int | None = None
    map_batches_failed: int | None = None
    threads_skipped: int | None = None
    reduce_fallback: bool = False
    fetch_seconds: float = 0.0
    clustering_seconds: float = 0.0
    within_thread_dedup_seconds: float = 0.0
    ranking_seconds: float = 0.0
    map_seconds: float = 0.0
    reduce_seconds: float = 0.0
    format_seconds: float = 0.0
    eval_seconds: float = 0.0
    notify_seconds: float = 0.0
    duration_seconds: float = 0.0
    error: str | None = None


def save_run_metrics(metrics: RunMetrics, root_dir: Path) -> Path:
    """Append one run-metrics JSON object to output/metrics.jsonl."""

    output_path = root_dir / "output" / "metrics.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(metrics)
    payload["date"] = metrics.generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
    payload["generated_at"] = metrics.generated_at.astimezone(timezone.utc).isoformat()
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path


def subtract_token_usage(
    total: dict[str, int] | None,
    partial: dict[str, int] | None,
) -> dict[str, int]:
    """Return total minus partial, clamped at zero for each field."""

    total_input = int((total or {}).get("input_tokens", 0))
    total_output = int((total or {}).get("output_tokens", 0))
    partial_input = int((partial or {}).get("input_tokens", 0))
    partial_output = int((partial or {}).get("output_tokens", 0))
    return {
        "input_tokens": max(0, total_input - partial_input),
        "output_tokens": max(0, total_output - partial_output),
    }
