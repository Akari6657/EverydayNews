"""Render final briefings to Markdown and optional structured JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import timezone
from pathlib import Path

from .models import (
    AppConfig,
    FinalBriefing,
    JsonOutputConfig,
    MarkdownOutputConfig,
    ThreadSummary,
)

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "briefing.md.j2"


def format_briefing(
    briefing: FinalBriefing,
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
    content = render_briefing(briefing, config, template_path)
    output_path.write_text(content, encoding="utf-8")
    if config.output.json.enabled:
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
    briefing: FinalBriefing,
    config: AppConfig,
    template_path: str | Path | None = None,
) -> str:
    """Render briefing content from the configured template."""

    template_file = Path(template_path) if template_path else DEFAULT_TEMPLATE_PATH
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")
    context = _build_context(briefing, config)
    return _render_template(template_file, context)


def _build_context(
    briefing: FinalBriefing,
    config: AppConfig,
) -> dict[str, str | int]:
    """Build the template context dictionary."""

    return {
        "date": briefing.date,
        "generated_at": briefing.generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_names": _source_names_for_briefing(briefing, config),
        "article_count": briefing.total_articles,
        "thread_count": briefing.total_threads,
        "source_count": briefing.total_sources,
        "llm_content": _render_structured_markdown(briefing),
        "model": briefing.model,
        "input_tokens": briefing.token_usage.get("input_tokens", 0),
        "output_tokens": briefing.token_usage.get("output_tokens", 0),
    }


def _render_template(template_file: Path, context: dict[str, str | int]) -> str:
    """Render the briefing template with Jinja2."""

    from jinja2 import Template as JinjaTemplate

    raw_template = template_file.read_text(encoding="utf-8")
    return JinjaTemplate(raw_template).render(**context).strip() + "\n"


def _source_names_for_briefing(
    briefing: FinalBriefing,
    config: AppConfig,
) -> str:
    """Return the sources displayed in the template header."""

    names = _collect_briefing_sources(briefing)
    if names:
        return "、".join(names)
    return "、".join(source.name for source in config.sources)


def _collect_briefing_sources(briefing: FinalBriefing) -> list[str]:
    """Collect distinct source names in stable order from a structured briefing."""

    names: list[str] = []
    for item in briefing.all_stories:
        for source_name in item.source_names:
            if source_name not in names:
                names.append(source_name)
    return names


def _render_structured_markdown(briefing: FinalBriefing) -> str:
    """Render FinalBriefing content to a Markdown body string."""

    lines: list[str] = [
        "## 今日综述",
        "",
        briefing.overview_zh.strip(),
    ]
    if briefing.top_stories:
        lines.extend(["", "---", "", "## 🔥 今日头条", ""])
        for item in briefing.top_stories:
            lines.extend(_top_story_lines(item))
            lines.append("")
    if briefing.other_stories:
        lines.extend(["---", "", "## 📎 其他值得关注", ""])
        for item in briefing.other_stories:
            lines.extend(_other_story_lines(item))
    return "\n".join(lines).strip()


def _top_story_lines(item: ThreadSummary) -> list[str]:
    """Render one top story into Markdown lines."""

    lines = [
        f"### {item.headline_zh}",
        item.summary_zh,
        f"> 来源：{' · '.join(item.source_names)} · [原文]({item.primary_link}) · 重要性 {item.importance}/10",
    ]
    if item.entities:
        lines.insert(2, f"关键实体：{'、'.join(item.entities)}")
    return lines


def _other_story_lines(item: ThreadSummary) -> list[str]:
    """Render one secondary story as a compact bullet."""

    return [
        f"- **{item.headline_zh}** — {item.summary_zh} _（{'、'.join(item.source_names)}）_ · [原文]({item.primary_link})",
    ]


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
