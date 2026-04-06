"""Render final briefing Markdown and save it to disk."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
from string import Template

from .models import AppConfig, Article, BriefingResult

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "briefing.md.j2"


def format_briefing(
    briefing: BriefingResult,
    articles: list[Article],
    config: AppConfig,
    template_path: str | Path | None = None,
) -> Path:
    """Render and write the final Markdown briefing."""

    output_dir = config.root_dir / config.output.markdown.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    content = render_briefing(briefing, articles, config, template_path)
    filename = briefing.generated_at.astimezone(timezone.utc).strftime("briefing-%Y-%m-%d.md")
    output_path = output_dir / filename
    output_path.write_text(content, encoding="utf-8")
    return output_path


def render_briefing(
    briefing: BriefingResult,
    articles: list[Article],
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
    briefing: BriefingResult,
    articles: list[Article],
    config: AppConfig,
) -> dict[str, str | int]:
    """Build the template context dictionary."""

    source_names = "、".join(source.name for source in config.sources)
    return {
        "generated_at": briefing.generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_names": source_names,
        "article_count": len(articles),
        "llm_content": briefing.content,
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
