"""Tests for Markdown briefing rendering."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.formatter import format_briefing, render_briefing
from src.models import FinalBriefing, ThreadSummary


def _make_briefing() -> FinalBriefing:
    """Create a minimal structured briefing for formatter tests."""

    return FinalBriefing(
        date="2026-04-06",
        overview_zh="今日综述。",
        topics={
            "国际政治": [
                ThreadSummary(
                    thread_id="thread-1",
                    topic="国际政治",
                    headline_zh="伊朗局势升级",
                    summary_zh="中东局势持续紧张。",
                    importance=9,
                    entities=["伊朗", "美国"],
                    source_names=["New York Times", "BBC News"],
                    primary_link="https://example.com/story",
                )
            ]
        },
        total_threads=1,
        total_sources=2,
        generated_at=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 15, "output_tokens": 30},
        model="deepseek-chat",
    )


def test_format_briefing_writes_output_file(sample_config, tmp_path) -> None:
    """Formatter should render and save the final Markdown file."""

    template_path = tmp_path / "briefing.md.j2"
    template_path.write_text("{{ llm_content }}\n{{ article_count }}\n", encoding="utf-8")
    briefing = _make_briefing()

    output_path = format_briefing(briefing, None, sample_config, template_path=template_path)

    assert output_path.exists()
    assert output_path.parent.name == "2026-04"
    assert output_path.parent.parent.name == "md"
    assert "### 伊朗局势升级" in output_path.read_text(encoding="utf-8")


def test_render_briefing_raises_for_missing_template(sample_config) -> None:
    """Formatter should fail clearly when the template file is missing."""

    briefing = _make_briefing()

    with pytest.raises(FileNotFoundError):
        render_briefing(briefing, None, sample_config, template_path="missing-template.j2")


def test_format_briefing_writes_structured_json_for_final_briefing(sample_config, tmp_path) -> None:
    """Formatter should write a JSON companion file for structured briefings."""

    template_path = tmp_path / "briefing.md.j2"
    template_path.write_text("{{ llm_content }}\n{{ article_count }}\n", encoding="utf-8")
    briefing = _make_briefing()

    output_path = format_briefing(briefing, None, sample_config, template_path=template_path)

    json_path = output_path.with_suffix(".json")
    assert output_path.exists()
    assert output_path.parent.name == "2026-04"
    assert output_path.parent.parent.name == "md"
    assert json_path.exists() is False
    json_path = sample_config.root_dir / "output" / "json" / "2026-04" / "briefing-2026-04-06.json"
    assert json_path.exists()
    markdown = output_path.read_text(encoding="utf-8")
    assert "### 伊朗局势升级" in markdown
    assert "1" in markdown
    payload = json_path.read_text(encoding="utf-8")
    assert "\"overview_zh\": \"今日综述。\"" in payload


def test_render_briefing_supports_final_briefing(sample_config, tmp_path) -> None:
    """Structured briefings should render to Markdown content."""

    template_path = tmp_path / "briefing.md.j2"
    template_path.write_text("{{ llm_content }}", encoding="utf-8")
    briefing = FinalBriefing(
        date="2026-04-06",
        overview_zh="今天重点关注国际政治与能源市场。",
        topics={
            "经济金融": [
                ThreadSummary(
                    thread_id="thread-2",
                    topic="经济金融",
                    headline_zh="油价波动加剧",
                    summary_zh="国际油价因地缘政治变化而震荡。",
                    importance=7,
                    entities=["布伦特原油"],
                    source_names=["BBC News"],
                    primary_link="https://example.com/oil",
                )
            ]
        },
        total_threads=1,
        total_sources=1,
        generated_at=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 0, "output_tokens": 0},
        model="deepseek-chat",
    )

    content = render_briefing(briefing, None, sample_config, template_path=template_path)

    assert "今天重点关注国际政治与能源市场。" in content
    assert "## 经济金融" in content
    assert "- 链接：https://example.com/oil" in content
