"""Tests for briefing evaluation and run metrics persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from src.evaluator import append_run_metrics, evaluate_briefing, write_evaluation_result
from src.models import ClusterSummary, EvaluationConfig, EvaluationResult, RunMetrics


@dataclass
class FakeMessage:
    """Fake assistant message."""

    content: str


@dataclass
class FakeChoice:
    """Fake response choice wrapper."""

    message: FakeMessage


@dataclass
class FakeUsage:
    """Fake token usage payload."""

    prompt_tokens: int = 12
    completion_tokens: int = 34


@dataclass
class FakeResponse:
    """Fake OpenAI-compatible response."""

    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)
    model: str = "deepseek-chat"


class FakeCompletions:
    """Fake chat completions endpoint."""

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
    """Fake OpenAI client with nested chat endpoints."""

    def __init__(self, responses):
        self.chat = type("Chat", (), {"completions": FakeCompletions(responses)})()


def _evaluation_response() -> FakeResponse:
    """Return one valid JSON evaluation response."""

    payload = {
        "coverage": 8,
        "diversity": 7,
        "clarity": 9,
        "redundancy": 8,
        "importance_calibration": 7,
        "notes": "整体质量稳定。",
    }
    return FakeResponse(choices=[FakeChoice(FakeMessage(json.dumps(payload, ensure_ascii=False)))])


def test_evaluate_briefing_parses_scores(sample_config) -> None:
    """Evaluator should parse model JSON into an EvaluationResult."""

    config = replace(
        sample_config,
        evaluation=EvaluationConfig(enabled=True, max_retries=2),
    )
    summary = ClusterSummary(
        cluster_id="cluster-1",
        topic="国际政治",
        headline_zh="测试标题",
        summary_zh="测试摘要",
        importance=8,
        entities=["伊朗"],
        source_names=["New York Times"],
        primary_link="https://example.com/story",
    )
    client = FakeClient([_evaluation_response()])

    result = evaluate_briefing(
        "# 简报\n\n测试内容",
        [summary],
        config,
        client=client,
        now=datetime(2026, 4, 7, 12, 0, tzinfo=timezone.utc),
    )

    assert result.scores == {
        "coverage": 8,
        "diversity": 7,
        "clarity": 9,
        "redundancy": 8,
        "importance_calibration": 7,
    }
    assert result.notes == "整体质量稳定。"
    assert result.token_usage == {"input_tokens": 12, "output_tokens": 34}


def test_evaluate_briefing_retries_invalid_json(sample_config, monkeypatch) -> None:
    """Evaluator should retry once when the first response is invalid JSON."""

    monkeypatch.setattr("src.evaluator.time.sleep", lambda _: None)
    config = replace(
        sample_config,
        evaluation=EvaluationConfig(enabled=True, max_retries=2),
    )
    client = FakeClient(
        [
            FakeResponse(choices=[FakeChoice(FakeMessage("not-json"))]),
            _evaluation_response(),
        ]
    )

    result = evaluate_briefing("# 简报", [], config, client=client)

    assert len(client.chat.completions.calls) == 2
    assert result.coverage == 8


def test_evaluate_briefing_raises_clear_message_for_content_risk(sample_config) -> None:
    """Content-risk blocking should produce a clear skip message."""

    config = replace(
        sample_config,
        evaluation=EvaluationConfig(enabled=True, max_retries=1),
    )
    client = FakeClient([RuntimeError("Error code: 400 - Content Exists Risk")])

    try:
        evaluate_briefing("# 简报", [], config, client=client)
    except RuntimeError as exc:
        assert "content risk" in str(exc).casefold()
        return
    raise AssertionError("Expected RuntimeError for content risk")


def test_write_evaluation_and_metrics_files(sample_config) -> None:
    """Evaluator should persist evaluation JSON and append JSONL metrics."""

    generated_at = datetime(2026, 4, 7, 12, 0, tzinfo=timezone.utc)
    evaluation = EvaluationResult(
        coverage=8,
        diversity=7,
        clarity=9,
        redundancy=8,
        importance_calibration=7,
        notes="整体质量稳定。",
        token_usage={"input_tokens": 12, "output_tokens": 34},
        model="deepseek-chat",
        generated_at=generated_at,
    )
    eval_path = write_evaluation_result(evaluation, generated_at, sample_config)
    metrics = RunMetrics(
        date="2026-04-07",
        articles_fetched=79,
        clusters=30,
        map_summaries_generated=25,
        after_importance_filter=12,
        final_items=10,
        total_tokens=12345,
        duration_seconds=42.1234,
        map_batches_total=6,
        map_batches_failed=1,
        map_clusters_skipped=5,
        eval_scores=evaluation.scores,
        eval_notes=evaluation.notes,
    )
    metrics_path = append_run_metrics(metrics, sample_config)

    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    assert eval_path.name == "briefing-2026-04-07.eval.json"
    assert eval_payload["coverage"] == 8
    assert eval_payload["model"] == "deepseek-chat"

    metrics_lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert metrics_path.name == "metrics.jsonl"
    assert len(metrics_lines) == 1
    metrics_payload = json.loads(metrics_lines[0])
    assert metrics_payload["clusters"] == 30
    assert metrics_payload["map_clusters_skipped"] == 5
    assert metrics_payload["eval_scores"]["clarity"] == 9
