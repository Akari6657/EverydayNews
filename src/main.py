"""CLI entry point for the Daily Headline Agent pipeline."""

from __future__ import annotations

import argparse
import logging
from time import perf_counter
from typing import Sequence

from .config_loader import get_config
from .dedup import deduplicate
from .evaluator import append_run_metrics, evaluate_briefing, write_evaluation_result
from .fetcher import fetch_all_feeds
from .formatter import format_briefing
from .models import ArticleCluster, RunMetrics
from .notifier import notify
from .summarizer_map import summarize_clusters_with_usage
from .summarizer_reduce import build_final_briefing, count_reduce_candidates

LOGGER = logging.getLogger(__name__)


def run_pipeline(config_path: str = "config.yaml", dry_run: bool = False):
    """Run one pipeline invocation."""

    started_at = perf_counter()
    config = None
    articles = []
    clusters = []
    map_result = None
    reduce_candidates = 0
    briefing = None
    evaluation_result = None
    output_path = None
    current_stage = "config"
    failure_reason = None
    try:
        config = get_config(config_path)
        current_stage = "fetch"
        articles = fetch_all_feeds(config)
        LOGGER.info("Fetched %s articles", len(articles))
        current_stage = "dedup"
        clusters = deduplicate(articles, config)
        LOGGER.info("After dedup: %s clusters", len(clusters))
        if dry_run:
            _log_dry_run_clusters(clusters)
            return clusters
        current_stage = "map"
        map_result = summarize_clusters_with_usage(clusters, config)
        LOGGER.info(
            "Map-stage summaries generated: %s clusters, tokens: %s",
            len(map_result.summaries),
            map_result.token_usage,
        )
        reduce_candidates = count_reduce_candidates(map_result.summaries, config)
        current_stage = "reduce"
        briefing = build_final_briefing(
            map_result.summaries,
            config,
            token_usage=map_result.token_usage,
        )
        LOGGER.info("Reduce-stage briefing generated, tokens: %s", briefing.token_usage)
        current_stage = "format"
        output_path = format_briefing(briefing, None, config)
        LOGGER.info("Briefing saved to %s", output_path)
        current_stage = "notify"
        notify(output_path, briefing, config)
        current_stage = "evaluation"
        if config.evaluation.enabled:
            try:
                markdown_content = output_path.read_text(encoding="utf-8")
                evaluation_result = evaluate_briefing(markdown_content, map_result.summaries, config)
                evaluation_path = write_evaluation_result(evaluation_result, briefing.generated_at, config)
                LOGGER.info("Evaluation saved to %s", evaluation_path)
            except Exception as exc:
                if "content risk" in str(exc).casefold():
                    LOGGER.warning(
                        "Evaluation skipped due to content risk; briefing remains valid and no .eval.json was written"
                    )
                else:
                    LOGGER.warning("Evaluation failed: %s", exc)
        return output_path
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        if config is not None and not dry_run:
            metrics = RunMetrics(
                date=_metrics_date(briefing),
                articles_fetched=len(articles),
                clusters=len(clusters),
                map_summaries_generated=len(map_result.summaries) if map_result else 0,
                after_importance_filter=reduce_candidates,
                final_items=briefing.total_clusters if briefing else 0,
                total_tokens=_metrics_total_tokens(briefing, map_result, evaluation_result),
                duration_seconds=perf_counter() - started_at,
                map_batches_total=map_result.batches_total if map_result else 0,
                map_batches_failed=map_result.batches_failed if map_result else 0,
                map_clusters_skipped=map_result.clusters_skipped if map_result else 0,
                eval_scores=evaluation_result.scores if evaluation_result else None,
                eval_notes=evaluation_result.notes if evaluation_result else None,
                status="failed" if failure_reason else "success",
                failure_stage=current_stage if failure_reason else None,
                failure_reason=failure_reason[:300] if failure_reason else None,
            )
            try:
                metrics_path = append_run_metrics(metrics, config)
                LOGGER.info("Metrics appended to %s", metrics_path)
            except Exception as exc:
                LOGGER.warning("Failed to append metrics log: %s", exc)


def run_scheduled(config_path: str = "config.yaml") -> None:
    """Run the pipeline on a daily APScheduler cron."""

    config = get_config(config_path)
    hour, minute = _parse_run_at(config.schedule.run_at)
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError as exc:
        raise RuntimeError("APScheduler is required for --schedule mode") from exc
    scheduler = BlockingScheduler(timezone=config.schedule.timezone)
    scheduler.add_job(run_pipeline, "cron", hour=hour, minute=minute, args=[config_path])
    LOGGER.info("Scheduled daily run at %s (%s)", config.schedule.run_at, config.schedule.timezone)
    scheduler.start()


def _parse_run_at(value: str) -> tuple[int, int]:
    """Parse a HH:MM schedule string."""

    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except ValueError as exc:
        raise ValueError("schedule.run_at must use HH:MM format") from exc
    if hour not in range(24) or minute not in range(60):
        raise ValueError("schedule.run_at must be a valid 24-hour time")
    return hour, minute


def _log_dry_run_clusters(clusters: list[ArticleCluster]) -> None:
    """Log dry-run cluster summaries."""

    if not clusters:
        LOGGER.info("Dry run complete: no new articles found")
        return
    for index, cluster in enumerate(clusters, start=1):
        published = cluster.primary.published.strftime("%Y-%m-%d %H:%M UTC")
        LOGGER.info(
            "[%s] %s 家来源 | %s | %s | %s",
            index,
            cluster.source_count,
            cluster.primary.source_name,
            published,
            cluster.primary.title,
        )


def _total_tokens(primary: dict[str, int], extra: dict[str, int] | None = None) -> int:
    """Return the combined token count across one or two usage payloads."""

    total = int(primary.get("input_tokens", 0)) + int(primary.get("output_tokens", 0))
    if extra:
        total += int(extra.get("input_tokens", 0)) + int(extra.get("output_tokens", 0))
    return total


def _metrics_total_tokens(briefing, map_result, evaluation_result) -> int:
    """Return total token usage accumulated so far in the pipeline."""

    if briefing is not None:
        return _total_tokens(
            briefing.token_usage,
            evaluation_result.token_usage if evaluation_result else None,
        )
    if map_result is not None:
        return _total_tokens(
            map_result.token_usage,
            evaluation_result.token_usage if evaluation_result else None,
        )
    if evaluation_result is not None:
        return _total_tokens(evaluation_result.token_usage)
    return 0


def _metrics_date(briefing) -> str:
    """Return the date written into run metrics records."""

    if briefing is not None:
        return briefing.date
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Daily Headline Agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--schedule", action="store_true", help="Run with APScheduler")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and deduplicate without LLM calls")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    _configure_logging()
    args = build_parser().parse_args(argv)
    if args.schedule:
        run_scheduled(args.config)
        return 0
    run_pipeline(args.config, dry_run=args.dry_run)
    return 0


def _configure_logging() -> None:
    """Set a consistent logging format."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    main()
