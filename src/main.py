"""CLI entry point for the Daily Headline Agent pipeline."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .config_loader import get_config
from .dedup import deduplicate
from .fetcher import fetch_all_feeds
from .formatter import format_briefing
from .models import ArticleCluster
from .notifier import notify
from .summarizer_map import summarize_clusters_with_usage
from .summarizer_reduce import build_final_briefing

LOGGER = logging.getLogger(__name__)


def run_pipeline(config_path: str = "config.yaml", dry_run: bool = False):
    """Run one pipeline invocation."""

    config = get_config(config_path)
    articles = fetch_all_feeds(config)
    LOGGER.info("Fetched %s articles", len(articles))
    clusters = deduplicate(articles, config)
    LOGGER.info("After dedup: %s clusters", len(clusters))
    if dry_run:
        _log_dry_run_clusters(clusters)
        return clusters
    map_result = summarize_clusters_with_usage(clusters, config)
    LOGGER.info(
        "Map-stage summaries generated: %s clusters, tokens: %s",
        len(map_result.summaries),
        map_result.token_usage,
    )
    briefing = build_final_briefing(
        map_result.summaries,
        config,
        token_usage=map_result.token_usage,
    )
    LOGGER.info("Reduce-stage briefing generated, tokens: %s", briefing.token_usage)
    output_path = format_briefing(briefing, None, config)
    LOGGER.info("Briefing saved to %s", output_path)
    notify(output_path, briefing, config)
    return output_path


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
