"""Render final briefings to Markdown and optional structured JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import timezone
from pathlib import Path
from string import Template

from .models import (
    AppConfig,
    Article,
    BriefingResult,
    ClusterSummary,
    FinalBriefing,
    JsonOutputConfig,
    MarkdownOutputConfig,
)

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "briefing.md.j2"


def format_briefing(
    briefing: BriefingResult | FinalBriefing,
    articles: list[Article] | None,
    config: AppConfig,
    template_path: str | Path | None = None,
) -> Path:
    """Render and write the final Markdown briefing."""

    output_path = _resolve_output_path(
        config.root_dir,
        config.output.markdown,
        briefing.generated_at,
        "md",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = render_briefing(briefing, articles, config, template_path)
    output_path.write_text(content, encoding="utf-8")
    if isinstance(briefing, FinalBriefing) and config.output.json.enabled:
        json_path = _resolve_output_path(
            config.root_dir,
            config.output.json,
            briefing.generated_at,
            "json",
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(_render_structured_json(briefing), encoding="utf-8")
    return output_path


def render_briefing(
    briefing: BriefingResult | FinalBriefing,
    articles: list[Article] | None,
    config: AppConfig,
    template_path: str | Path | None = None,
) -> str:
    """Render briefing content from the configured template."""

    template_file = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    context = _build_context(briefing, articles, config)
    return _render_template(template_file, context)


def _build_context(
    briefing: BriefingResult | FinalBriefing,
    articles: list[Article] | None,
    config: AppConfig,
) -> dict[str, str | int]:
    """Build the template context dictionary."""

    source_names = _source_names_for_briefing(briefing, config)
    return {
        "generated_at": briefing.generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_names": source_names,
        "article_count": _article_count_for_briefing(briefing, articles),
        "llm_content": _content_for_briefing(briefing),
        "model": briefing.model,
        "input_tokens": briefing.token_usage.get("input_tokens", 0),
        "output_tokens": briefing.token_usage.get("output_tokens", 0),
    }


def _render_template(template_file: Path, context: dict[str, str | int]) -> str:
    """Render Jinja2 when available, with a simple fallback."""

    raw_template = template_file.read_text(encoding="utf-8")
    try:
        from jinja2 import Template as JinjaTemplate

        return JinjaTemplate(raw_template).render(**context).strip() + "\n"
    except ImportError:
        sanitized = raw_template
        for key in context:
            sanitized = sanitized.replace(f"{{{{ {key} }}}}", f"${key}")
        return Template(sanitized).safe_substitute(context).strip() + "\n"


def _content_for_briefing(briefing: BriefingResult | FinalBriefing) -> str:
    """Return the Markdown body content for the given briefing type."""

    if isinstance(briefing, FinalBriefing):
        return _render_structured_markdown(briefing)
    return briefing.content


def _article_count_for_briefing(
    briefing: BriefingResult | FinalBriefing,
    articles: list[Article] | None,
) -> int:
    """Return the count displayed in the template header."""

    if isinstance(briefing, FinalBriefing):
        return briefing.total_clusters
    return len(articles or [])


def _source_names_for_briefing(
    briefing: BriefingResult | FinalBriefing,
    config: AppConfig,
) -> str:
    """Return the sources displayed in the template header."""

    if isinstance(briefing, FinalBriefing):
        names = _collect_briefing_sources(briefing)
        if names:
            return "、".join(names)
    return "、".join(source.name for source in config.sources)


def _collect_briefing_sources(briefing: FinalBriefing) -> list[str]:
    """Collect distinct source names in stable order from a structured briefing."""

    names: list[str] = []
    for items in briefing.topics.values():
        for item in items:
            for source_name in item.source_names:
                if source_name not in names:
                    names.append(source_name)
    return names


def _render_structured_markdown(briefing: FinalBriefing) -> str:
    """Render FinalBriefing content to a Markdown body string."""

    lines: list[str] = [briefing.overview_zh.strip()]
    if briefing.topics:
        lines.append("")
    for topic_name, items in briefing.topics.items():
        lines.append(f"## {topic_name}")
        lines.append("")
        for item in items:
            lines.extend(_cluster_summary_lines(item))
            lines.append("")
    return "\n".join(line for line in lines if line is not None).strip()


def _cluster_summary_lines(item: ClusterSummary) -> list[str]:
    """Render one structured summary into Markdown lines."""

    lines = [
        f"### {item.headline_zh}",
        item.summary_zh,
        f"- 来源：{' / '.join(item.source_names)}",
        f"- 链接：{item.primary_link}",
        f"- 重要性：{item.importance}/10",
    ]
    if item.entities:
        lines.append(f"- 实体：{'、'.join(item.entities)}")
    return lines


def _render_structured_json(briefing: FinalBriefing) -> str:
    """Render FinalBriefing into a JSON document."""

    payload = asdict(briefing)
    payload["generated_at"] = briefing.generated_at.astimezone(timezone.utc).isoformat()
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _resolve_output_path(
    root_dir: Path,
    output_config: MarkdownOutputConfig | JsonOutputConfig,
    generated_at,
    suffix: str,
) -> Path:
    """Build an output file path with optional monthly partitioning."""

    timestamp = generated_at.astimezone(timezone.utc)
    output_dir = root_dir / output_config.directory
    if output_config.group_by_month:
        output_dir = output_dir / timestamp.strftime("%Y-%m")
    filename = timestamp.strftime(f"briefing-%Y-%m-%d.{suffix}")
    return output_dir / filename
