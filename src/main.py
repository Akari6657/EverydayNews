"""CLI entry point for the Daily Headline Agent pipeline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import logging
from time import perf_counter
from typing import Sequence

from .config_loader import get_config
from .dedup import deduplicate, deduplicate_with_diagnostics
from .evaluator import append_run_metrics, evaluate_briefing, write_evaluation_result
from .fetcher import fetch_all_feeds
from .formatter import format_briefing
from .models import AppConfig, Article, ArticleCluster, DedupDiagnostics, RunMetrics
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
        if dry_run:
            clusters, dedup_diagnostics = deduplicate_with_diagnostics(articles, config)
        else:
            dedup_diagnostics = None
            clusters = deduplicate(articles, config)
        LOGGER.info("After dedup: %s clusters", len(clusters))
        if dry_run:
            _log_dry_run_fetch_overview(config, articles)
            _log_dry_run_dedup_overview(config, dedup_diagnostics, clusters)
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
            "[%s] %s source(s) | %s article(s) | primary=%s/%s | %s | %s",
            index,
            cluster.source_count,
            len(cluster.all_articles),
            cluster.primary.source_name,
            cluster.primary.category,
            published,
            cluster.primary.title,
        )
        LOGGER.info("    sources: %s", " / ".join(cluster.source_names))
        for position, article in enumerate(cluster.all_articles, start=1):
            marker = "*" if position == 1 else "-"
            LOGGER.info(
                "    %s %s/%s | %s",
                marker,
                article.source_name,
                article.category,
                article.title,
            )


def _log_dry_run_fetch_overview(config: AppConfig, articles: list[Article]) -> None:
    """Log how many post-filter articles each configured source contributed."""

    total_feeds = sum(len(source.feeds) for source in config.sources)
    LOGGER.info(
        "Dry-run fetch overview | sources=%s | feeds=%s | retained_articles=%s | max_articles_per_source=%s",
        len(config.sources),
        total_feeds,
        len(articles),
        config.pipeline.max_articles_per_source,
    )
    counts_by_source = Counter(article.source_slug for article in articles)
    categories_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    latest_by_source: dict[str, Article] = {}
    for article in articles:
        categories_by_source[article.source_slug][article.category] += 1
        current_latest = latest_by_source.get(article.source_slug)
        if current_latest is None or article.published > current_latest.published:
            latest_by_source[article.source_slug] = article
    for source in config.sources:
        category_counts = categories_by_source.get(source.slug, Counter())
        category_summary = ", ".join(
            f"{category}={count}" for category, count in category_counts.most_common()
        ) or "none"
        LOGGER.info(
            "Fetch kept | %s | articles=%s | categories=%s | latest=%s",
            source.name,
            counts_by_source.get(source.slug, 0),
            category_summary,
            latest_by_source[source.slug].published.strftime("%Y-%m-%d %H:%M UTC")
            if source.slug in latest_by_source
            else "n/a",
        )


def _log_dry_run_dedup_overview(
    config: AppConfig,
    diagnostics: DedupDiagnostics | None,
    clusters: list[ArticleCluster],
) -> None:
    """Log how the fetched article pool collapsed into deduplicated clusters."""

    if diagnostics is None:
        return
    represented_articles = sum(len(cluster.all_articles) for cluster in clusters)
    LOGGER.info(
        "Dry-run dedup overview | method=%s | similarity_threshold=%.2f | seen_filtered=%s | fresh_articles=%s | clusters_before_limit=%s | clusters_after_limit=%s | multi_source_clusters=%s | represented_articles=%s",
        config.dedup.method,
        config.dedup.similarity_threshold,
        diagnostics.seen_filtered,
        diagnostics.fresh_articles,
        diagnostics.clusters_before_limit,
        diagnostics.clusters_after_limit,
        diagnostics.multi_source_clusters,
        represented_articles,
    )
    truncated = diagnostics.clusters_before_limit - diagnostics.clusters_after_limit
    if truncated > 0:
        LOGGER.info(
            "Dry-run dedup cap applied | kept_top_clusters=%s | truncated_clusters=%s | cap=%s",
            diagnostics.clusters_after_limit,
            truncated,
            config.pipeline.total_articles_for_summary,
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
    for logger_name in ("httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
