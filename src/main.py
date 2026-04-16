"""CLI entry point for the Daily Headline Agent pipeline."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Sequence

from .config_loader import get_config
from .dedup import deduplicate_within_thread_with_diagnostics
from .fetcher import fetch_all_feeds
from .formatter import format_briefing
from .metrics import RunMetrics, save_run_metrics, subtract_token_usage
from .models import (
    AppConfig,
    StoryThread,
    ThreadDedupDiagnostics,
)
from .notifier import notify
from .ranker import rank_threads
from .summarizer_map import summarize_threads_with_usage
from .summarizer_reduce import build_final_briefing, count_reduce_candidates
from .thread_clusterer import cluster_into_threads

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    config_path: str = "config.yaml",
    dry_run: bool = False,
    dump_threads: bool = False,
    dedup_within_threads: bool = False,
    eval: bool = False,
):
    """Run one pipeline invocation."""

    run_started = time.perf_counter()
    config: AppConfig | None = None
    metrics_fields: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc),
        "mode": _run_mode(dry_run=dry_run, dump_threads=dump_threads),
        "success": False,
        "config_path": config_path,
        "eval_enabled": eval,
    }

    try:
        config = get_config(config_path)
        metrics_fields["config_path"] = str(config.config_path)

        stage_started = time.perf_counter()
        articles = fetch_all_feeds(config)
        metrics_fields["fetch_seconds"] = _elapsed_seconds(stage_started)
        metrics_fields["fetched_articles"] = len(articles)
        LOGGER.info("Fetched %s articles", len(articles))

        if dump_threads:
            stage_started = time.perf_counter()
            threads = cluster_into_threads(articles, config)
            metrics_fields["clustering_seconds"] = _elapsed_seconds(stage_started)
            metrics_fields["threads_after_clustering"] = len(threads)
            thread_diagnostics = None
            if dedup_within_threads:
                stage_started = time.perf_counter()
                threads, thread_diagnostics = _dedup_threads_for_dump(threads, config)
                metrics_fields["within_thread_dedup_seconds"] = _elapsed_seconds(stage_started)
                metrics_fields["threads_after_within_thread_dedup"] = len(threads)
                metrics_fields["within_thread_changed_threads"] = _changed_thread_count(thread_diagnostics)
                metrics_fields["within_thread_removed_articles"] = _removed_article_count(thread_diagnostics)
            _log_thread_dump(threads, thread_diagnostics)
            metrics_fields["success"] = True
            return threads

        stage_started = time.perf_counter()
        threads = cluster_into_threads(articles, config)
        metrics_fields["clustering_seconds"] = _elapsed_seconds(stage_started)
        metrics_fields["threads_after_clustering"] = len(threads)
        LOGGER.info("After thread clustering: %s threads", len(threads))

        stage_started = time.perf_counter()
        threads, thread_diagnostics = _dedup_threads_for_dump(threads, config)
        metrics_fields["within_thread_dedup_seconds"] = _elapsed_seconds(stage_started)
        metrics_fields["threads_after_within_thread_dedup"] = len(threads)
        metrics_fields["within_thread_changed_threads"] = _changed_thread_count(thread_diagnostics)
        metrics_fields["within_thread_removed_articles"] = _removed_article_count(thread_diagnostics)
        LOGGER.info(
            "After within-thread dedup: %s threads | changed_threads=%s | removed_articles=%s",
            len(threads),
            _changed_thread_count(thread_diagnostics),
            _removed_article_count(thread_diagnostics),
        )

        stage_started = time.perf_counter()
        threads = rank_threads(threads, config)
        metrics_fields["ranking_seconds"] = _elapsed_seconds(stage_started)
        metrics_fields["threads_after_ranking"] = len(threads)
        LOGGER.info("After thread ranking: %s threads", len(threads))
        if dry_run:
            _log_thread_dump(threads, thread_diagnostics)
            metrics_fields["success"] = True
            return threads

        stage_started = time.perf_counter()
        map_result = summarize_threads_with_usage(threads, config)
        metrics_fields["map_seconds"] = _elapsed_seconds(stage_started)
        metrics_fields["map_summaries"] = len(map_result.summaries)
        metrics_fields["map_token_usage"] = map_result.token_usage
        metrics_fields["map_batches_total"] = map_result.batches_total
        metrics_fields["map_batches_failed"] = map_result.batches_failed
        metrics_fields["threads_skipped"] = map_result.threads_skipped
        LOGGER.info(
            "Thread map-stage summaries generated: %s threads, tokens: %s",
            len(map_result.summaries),
            map_result.token_usage,
        )

        reduce_candidates = count_reduce_candidates(map_result.summaries, config)
        metrics_fields["reduce_candidates"] = reduce_candidates
        stage_started = time.perf_counter()
        briefing = build_final_briefing(
            map_result.summaries,
            config,
            token_usage=map_result.token_usage,
        )
        metrics_fields["reduce_seconds"] = _elapsed_seconds(stage_started)
        metrics_fields["briefing_threads"] = briefing.total_threads
        metrics_fields["briefing_sources"] = briefing.total_sources
        metrics_fields["briefing_articles"] = briefing.total_articles
        metrics_fields["total_token_usage"] = briefing.token_usage
        metrics_fields["reduce_token_usage"] = subtract_token_usage(
            briefing.token_usage,
            map_result.token_usage,
        )
        metrics_fields["reduce_fallback"] = _is_reduce_fallback(briefing.model)
        LOGGER.info(
            "Reduce-stage briefing generated | candidates=%s | tokens=%s",
            reduce_candidates,
            briefing.token_usage,
        )

        stage_started = time.perf_counter()
        output_path = format_briefing(briefing, config)
        metrics_fields["format_seconds"] = _elapsed_seconds(stage_started)
        metrics_fields["output_path"] = str(output_path)
        LOGGER.info("Briefing saved to %s", output_path)

        if eval:
            from .evaluator import evaluate_briefing, save_eval_result

            stage_started = time.perf_counter()
            eval_result = evaluate_briefing(briefing, map_result.summaries, config)
            eval_path = save_eval_result(eval_result, config)
            metrics_fields["eval_seconds"] = _elapsed_seconds(stage_started)
            metrics_fields["eval_output_path"] = str(eval_path)
            metrics_fields["eval_token_usage"] = eval_result.token_usage
            LOGGER.info(
                "Briefing eval | coverage=%s | diversity=%s | clarity=%s"
                " | redundancy=%s | importance_calibration=%s | notes=%s",
                eval_result.coverage,
                eval_result.diversity,
                eval_result.clarity,
                eval_result.redundancy,
                eval_result.importance_calibration,
                eval_result.notes,
            )
            LOGGER.info("Eval result saved to %s", eval_path)

        stage_started = time.perf_counter()
        notify(output_path, briefing, config)
        metrics_fields["notify_seconds"] = _elapsed_seconds(stage_started)
        metrics_fields["success"] = True
        return output_path
    except Exception as exc:
        metrics_fields["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metrics_fields["duration_seconds"] = _elapsed_seconds(run_started)
        if config is not None:
            _save_run_metrics_safely(config, metrics_fields)


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


def _log_thread_dump(
    threads: list[StoryThread],
    thread_diagnostics: dict[int, ThreadDedupDiagnostics] | None = None,
) -> None:
    """Log story-thread assignments for manual clustering review."""

    if not threads:
        LOGGER.info("Thread dump complete: no articles available")
        return
    LOGGER.info(
        "Thread dump overview | threads=%s | multi_source_threads=%s | singleton_threads=%s | max_articles_in_thread=%s",
        len(threads),
        sum(1 for thread in threads if thread.source_count >= 2),
        sum(1 for thread in threads if len(thread.articles) == 1),
        max(len(thread.articles) for thread in threads),
    )
    if thread_diagnostics:
        LOGGER.info(
            "Thread dump within-thread dedup | changed_threads=%s | removed_articles=%s",
            sum(1 for item in thread_diagnostics.values() if item.before_articles != item.after_articles),
            sum(item.before_articles - item.after_articles for item in thread_diagnostics.values()),
        )
    for thread in threads:
        diagnostics = thread_diagnostics.get(thread.thread_id) if thread_diagnostics else None
        header = (
            f"Thread [{thread.thread_id}] {thread.topic} ({thread.topic_en}) | "
            f"{len(thread.articles)} articles | {thread.source_count} sources"
        )
        if diagnostics and diagnostics.before_articles != diagnostics.after_articles:
            header += f" | within-thread dedup {diagnostics.before_articles}->{diagnostics.after_articles}"
        LOGGER.info(
            header,
        )
        LOGGER.info("  rationale: %s", thread.rationale)
        if diagnostics:
            for merge in diagnostics.merged_pairs:
                LOGGER.info(
                    "  ~ merged %.2f | keep=%s/%s | drop=%s/%s",
                    merge.similarity,
                    merge.kept_article.source_name,
                    merge.kept_article.title,
                    merge.removed_article.source_name,
                    merge.removed_article.title,
                )
        for index, article in enumerate(thread.articles, start=1):
            marker = "*" if index == 1 else "-"
            LOGGER.info(
                "  %s %s/%s | %s",
                marker,
                article.source_name,
                article.category,
                article.title,
            )


def _dedup_threads_for_dump(
    threads: list[StoryThread],
    config: AppConfig,
) -> tuple[list[StoryThread], dict[int, ThreadDedupDiagnostics]]:
    """Apply experimental within-thread near-duplicate cleanup for thread dumps."""

    deduplicated_threads: list[StoryThread] = []
    diagnostics_by_original_id: dict[int, ThreadDedupDiagnostics] = {}
    for thread in threads:
        deduplicated, diagnostics = deduplicate_within_thread_with_diagnostics(thread, config)
        deduplicated_threads.append(deduplicated)
        diagnostics_by_original_id[thread.thread_id] = diagnostics
    return deduplicated_threads, diagnostics_by_original_id


def _run_mode(*, dry_run: bool, dump_threads: bool) -> str:
    """Return a stable mode label for metrics."""

    if dump_threads:
        return "dump_threads"
    if dry_run:
        return "dry_run"
    return "run"


def _changed_thread_count(thread_diagnostics: dict[int, ThreadDedupDiagnostics]) -> int:
    """Count how many threads changed during within-thread dedup."""

    return sum(
        1
        for item in thread_diagnostics.values()
        if item.before_articles != item.after_articles
    )


def _removed_article_count(thread_diagnostics: dict[int, ThreadDedupDiagnostics]) -> int:
    """Count how many articles were removed during within-thread dedup."""

    return sum(item.before_articles - item.after_articles for item in thread_diagnostics.values())


def _elapsed_seconds(started_at: float) -> float:
    """Return elapsed wall-clock seconds rounded for metrics output."""

    return round(time.perf_counter() - started_at, 4)


def _is_reduce_fallback(model: str) -> bool:
    """Return whether the final briefing used local reduce fallback assembly."""

    return model.endswith("(fallback)")


def _save_run_metrics_safely(config: AppConfig, metrics_fields: dict[str, object]) -> None:
    """Write metrics without letting observability failures break the pipeline."""

    try:
        metrics = RunMetrics(**metrics_fields)
        output_path = save_run_metrics(metrics, config.root_dir)
        LOGGER.info("Run metrics saved to %s", output_path)
    except Exception as exc:
        LOGGER.warning("Failed to write run metrics: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Daily Headline Agent")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--schedule", action="store_true", help="Run with APScheduler")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and deduplicate without LLM calls")
    parser.add_argument("--dump-threads", action="store_true", help="Fetch and print story-thread assignments")
    parser.add_argument(
        "--dedup-within-threads",
        action="store_true",
        help="With --dump-threads, apply strict within-thread near-duplicate cleanup",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run LLM quality evaluation after generating the briefing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    _configure_logging()
    args = build_parser().parse_args(argv)
    if args.dedup_within_threads and not args.dump_threads:
        raise SystemExit("--dedup-within-threads requires --dump-threads")
    if args.schedule:
        run_scheduled(args.config)
        return 0
    run_pipeline(
        args.config,
        dry_run=args.dry_run,
        dump_threads=args.dump_threads,
        dedup_within_threads=args.dedup_within_threads,
        eval=args.eval,
    )
    return 0


def _configure_logging() -> None:
    """Set a consistent logging format."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for logger_name in ("httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
