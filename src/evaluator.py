"""Evaluate generated briefings and persist per-run metrics."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AppConfig, ClusterSummary, EvaluationResult, RunMetrics
from .prompts import (
    EVALUATION_JSON_RETRY_SUFFIX,
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_USER_PROMPT_TEMPLATE,
)

LOGGER = logging.getLogger(__name__)


def evaluate_briefing(
    briefing_markdown: str,
    summaries: list[ClusterSummary],
    config: AppConfig,
    client: Any | None = None,
    now: datetime | None = None,
) -> EvaluationResult:
    """Run an LLM-based quality evaluation for a generated briefing."""

    llm_client = client or _create_client(config)
    prompt = EVALUATION_USER_PROMPT_TEMPLATE.format(
        briefing_markdown=briefing_markdown.strip(),
        cluster_summaries=_build_summaries_payload(summaries),
    )
    max_retries = config.evaluation.max_retries
    for attempt in range(max_retries):
        try:
            response = _request_evaluation(llm_client, config, prompt)
        except Exception as exc:
            if attempt == max_retries - 1:
                if _is_content_risk_error(exc):
                    raise RuntimeError("Evaluation skipped due to content risk") from exc
                raise RuntimeError("Failed to request evaluation after retries") from exc
            LOGGER.warning("Evaluation request attempt %s failed: %s", attempt + 1, exc)
            time.sleep(2**attempt)
            continue

        raw_text = _extract_response_text(response)
        try:
            payload = json.loads(raw_text)
            if not isinstance(payload, dict):
                raise TypeError("Evaluation JSON payload must be an object")
            return _parse_evaluation_result(
                payload,
                response=response,
                config=config,
                generated_at=now or datetime.now(timezone.utc),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError("Failed to parse evaluation JSON after retries") from exc
            LOGGER.warning("Invalid evaluation JSON on attempt %s: %s", attempt + 1, exc)
            prompt = prompt + EVALUATION_JSON_RETRY_SUFFIX
            time.sleep(2**attempt)

    raise RuntimeError("Evaluation request loop ended unexpectedly")


def write_evaluation_result(
    evaluation: EvaluationResult,
    briefing_generated_at: datetime,
    config: AppConfig,
) -> Path:
    """Write one evaluation JSON document next to the structured briefing output."""

    output_path = _resolve_eval_output_path(config.root_dir, briefing_generated_at, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **evaluation.scores,
        "notes": evaluation.notes,
        "token_usage": evaluation.token_usage,
        "model": evaluation.model,
        "generated_at": evaluation.generated_at.astimezone(timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def append_run_metrics(metrics: RunMetrics, config: AppConfig) -> Path:
    """Append one JSONL metrics record for the current run."""

    metrics_path = _resolve_metrics_path(config)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    record = asdict(metrics)
    record["duration_seconds"] = round(metrics.duration_seconds, 3)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return metrics_path


def _request_evaluation(client: Any, config: AppConfig, prompt: str) -> Any:
    """Send one evaluation request in JSON mode."""

    return client.chat.completions.create(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


def _parse_evaluation_result(
    payload: dict[str, Any],
    response: Any,
    config: AppConfig,
    generated_at: datetime,
) -> EvaluationResult:
    """Parse model JSON into an EvaluationResult dataclass."""

    return EvaluationResult(
        coverage=_coerce_score(payload.get("coverage"), "coverage"),
        diversity=_coerce_score(payload.get("diversity"), "diversity"),
        clarity=_coerce_score(payload.get("clarity"), "clarity"),
        redundancy=_coerce_score(payload.get("redundancy"), "redundancy"),
        importance_calibration=_coerce_score(
            payload.get("importance_calibration"),
            "importance_calibration",
        ),
        notes=_optional_string(payload.get("notes")),
        token_usage=_response_token_usage(response),
        model=str(getattr(response, "model", config.llm.model)),
        generated_at=generated_at,
    )


def _build_summaries_payload(summaries: list[ClusterSummary]) -> str:
    """Render cluster summaries into a compact evaluation payload."""

    if not summaries:
        return "无候选新闻。"
    return "\n".join(_summary_block(summary) for summary in summaries)


def _summary_block(summary: ClusterSummary) -> str:
    """Render one summary block for the evaluation prompt."""

    return "\n".join(
        [
            f"[cluster_id: {summary.cluster_id}]",
            f"主题: {summary.topic}",
            f"标题: {summary.headline_zh}",
            f"摘要: {summary.summary_zh}",
            f"重要性: {summary.importance}/10",
            f"来源: {', '.join(summary.source_names)}",
            "---",
        ]
    )


def _coerce_score(value: Any, field_name: str) -> int:
    """Convert one model score into a bounded integer."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Evaluation field '{field_name}' must be numeric")
    return max(1, min(10, int(round(float(value)))))


def _optional_string(value: Any) -> str:
    """Return a stripped string or an empty string."""

    if isinstance(value, str):
        return value.strip()
    return ""


def _resolve_eval_output_path(root_dir: Path, generated_at: datetime, config: AppConfig) -> Path:
    """Build the monthly-partitioned evaluation JSON path."""

    timestamp = generated_at.astimezone(timezone.utc)
    output_dir = root_dir / config.output.json.directory
    if config.output.json.group_by_month:
        output_dir = output_dir / timestamp.strftime("%Y-%m")
    filename = timestamp.strftime("briefing-%Y-%m-%d.eval.json")
    return output_dir / filename


def _resolve_metrics_path(config: AppConfig) -> Path:
    """Resolve the append-only metrics log path."""

    markdown_dir = config.root_dir / config.output.markdown.directory
    json_dir = config.root_dir / config.output.json.directory
    common_root = Path(os.path.commonpath([str(markdown_dir), str(json_dir)]))
    return common_root / "metrics.jsonl"


def _create_client(config: AppConfig) -> Any:
    """Create an OpenAI-compatible client for DeepSeek."""

    from openai import OpenAI

    api_key = os.getenv(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{config.llm.api_key_env}' is required")
    return OpenAI(api_key=api_key, base_url=config.llm.base_url)


def _extract_response_text(response: Any) -> str:
    """Extract the first text response from an SDK payload."""

    choices = getattr(response, "choices", [])
    if not choices:
        raise RuntimeError("LLM response did not include any choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _response_token_usage(response: Any) -> dict[str, int]:
    """Extract prompt/completion token counts from a response."""

    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def _is_content_risk_error(exc: Exception) -> bool:
    """Return whether an evaluation request hit provider content-risk blocking."""

    return "Content Exists Risk" in str(exc)
