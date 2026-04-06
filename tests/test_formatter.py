"""Tests for Markdown briefing rendering."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.formatter import format_briefing, render_briefing
from src.models import BriefingResult


def test_format_briefing_writes_output_file(sample_config, make_article, tmp_path) -> None:
    """Formatter should render and save the final Markdown file."""

    template_path = tmp_path / "briefing.md.j2"
    template_path.write_text("{{ llm_content }}\n{{ article_count }}\n", encoding="utf-8")
    briefing = BriefingResult(
        content="## 今日重点",
        model="deepseek-chat",
        token_usage={"input_tokens": 10, "output_tokens": 20},
        generated_at=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
    )

    output_path = format_briefing(briefing, [make_article()], sample_config, template_path=template_path)

    assert output_path.exists()
    assert "## 今日重点" in output_path.read_text(encoding="utf-8")


def test_render_briefing_raises_for_missing_template(sample_config, make_article) -> None:
    """Formatter should fail clearly when the template file is missing."""

    briefing = BriefingResult(
        content="content",
        model="deepseek-chat",
        token_usage={"input_tokens": 0, "output_tokens": 0},
        generated_at=datetime.now(timezone.utc),
    )

    with pytest.raises(FileNotFoundError):
        render_briefing(briefing, [make_article()], sample_config, template_path="missing-template.j2")
