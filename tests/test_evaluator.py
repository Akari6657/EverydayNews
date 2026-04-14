"""Tests for LLM-based briefing quality evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.evaluator import evaluate_briefing, save_eval_result
from src.models import FinalBriefing, ThreadSummary


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeUsage:
    prompt_tokens: int = 300
    completion_tokens: int = 80


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "deepseek-chat"


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.chat = type("Chat", (), {"completions": FakeCompletions(responses)})()


def _valid_eval_json() -> str:
    return json.dumps(
        {
            "coverage": 8,
            "diversity": 7,
            "clarity": 9,
            "redundancy": 8,
            "importance_calibration": 7,
            "notes": "整体质量良好",
        },
        ensure_ascii=False,
    )


def _make_briefing(tmp_path: Path) -> FinalBriefing:
    return FinalBriefing(
        date="2026-04-14",
        overview_zh="今日重点关注中东局势与全球经济动向。",
        top_stories=[_summary(1, importance=9)],
        other_stories=[_summary(2, importance=5)],
        total_threads=2,
        total_sources=3,
        total_articles=5,
        generated_at=datetime(2026, 4, 14, 8, 0, tzinfo=timezone.utc),
        token_usage={"input_tokens": 1000, "output_tokens": 200},
        model="deepseek-chat",
    )


def _summary(thread_id: int, importance: int = 7) -> ThreadSummary:
    return ThreadSummary(
        thread_id=thread_id,
        topic="国际政治",
        headline_zh="测试标题",
        summary_zh="测试摘要内容",
        importance=importance,
        entities=[],
        source_names=["New York Times"],
        primary_link=f"https://example.com/{thread_id}",
        source_count=1,
        article_count=1,
    )


def test_evaluate_briefing_returns_valid_scores(sample_config, tmp_path) -> None:
    """Valid LLM response should produce an EvalResult with correct scores."""

    from dataclasses import replace
    config = replace(sample_config, root_dir=tmp_path)
    briefing = _make_briefing(tmp_path)
    all_summaries = [_summary(1, importance=9), _summary(2, importance=5)]
    client = FakeClient([FakeResponse(choices=[FakeChoice(FakeMessage(_valid_eval_json()))])])

    result = evaluate_briefing(
        briefing,
        all_summaries,
        config,
        client=client,
        now=datetime(2026, 4, 14, 8, 5, tzinfo=timezone.utc),
    )

    assert result.coverage == 8
    assert result.diversity == 7
    assert result.clarity == 9
    assert result.redundancy == 8
    assert result.importance_calibration == 7
    assert result.notes == "整体质量良好"
    assert result.date == "2026-04-14"
    assert result.candidate_count == 2
    assert result.briefing_thread_count == 2
    assert result.token_usage["input_tokens"] == 300
    assert result.token_usage["output_tokens"] == 80
    assert client.chat.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_evaluate_briefing_retries_invalid_json(sample_config, tmp_path, monkeypatch) -> None:
    """Invalid JSON on first attempt should trigger one retry."""

    monkeypatch.setattr("src.evaluator.time.sleep", lambda _: None)
    from dataclasses import replace
    config = replace(sample_config, root_dir=tmp_path)
    briefing = _make_briefing(tmp_path)
    client = FakeClient(
        [
            FakeResponse(choices=[FakeChoice(FakeMessage("not-json"))]),
            FakeResponse(choices=[FakeChoice(FakeMessage(_valid_eval_json()))]),
        ]
    )

    result = evaluate_briefing(briefing, [_summary(1)], config, client=client)

    assert len(client.chat.completions.calls) == 2
    assert result.coverage == 8
    assert result.notes == "整体质量良好"


def test_save_eval_result_writes_json_file(sample_config, tmp_path) -> None:
    """save_eval_result should write a valid JSON file under output/eval/."""

    from dataclasses import replace
    from src.models import EvalResult

    config = replace(sample_config, root_dir=tmp_path)
    result = EvalResult(
        date="2026-04-14",
        coverage=8,
        diversity=7,
        clarity=9,
        redundancy=8,
        importance_calibration=7,
        notes="测试评估",
        model="deepseek-chat",
        generated_at=datetime(2026, 4, 14, 8, 5, tzinfo=timezone.utc),
        token_usage={"input_tokens": 300, "output_tokens": 80},
        candidate_count=10,
        briefing_thread_count=5,
    )

    output_path = save_eval_result(result, config)

    assert output_path.exists()
    assert output_path.parent.name == "eval"
    assert output_path.name == "eval-2026-04-14.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["coverage"] == 8
    assert payload["notes"] == "测试评估"
    assert payload["candidate_count"] == 10
    assert payload["briefing_thread_count"] == 5
    assert "generated_at" in payload
