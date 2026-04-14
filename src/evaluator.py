"""LLM-based quality evaluation for generated daily briefings."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm_utils import (
    create_client,
    extract_response_text,
    load_json_payload,
    response_token_usage,
)
from .models import AppConfig, EvalResult, FinalBriefing, ThreadSummary
from .prompts import (
    EVALUATION_JSON_RETRY_SUFFIX,
    EVALUATION_SYSTEM_PROMPT,
    EVALUATION_USER_PROMPT_TEMPLATE,
)

LOGGER = logging.getLogger(__name__)

EVAL_SCORE_KEYS = ("coverage", "diversity", "clarity", "redundancy", "importance_calibration")


def evaluate_briefing(
    briefing: FinalBriefing,
    all_summaries: list[ThreadSummary],
    config: AppConfig,
    client: Any | None = None,
    now: datetime | None = None,
) -> EvalResult:
    """Grade the daily briefing with a 5-dimension LLM quality assessment.

    Uses the full map-stage candidate list (not just selected stories) so that
    the coverage score reflects whether important stories were incorrectly filtered.
    """

    generated_at = now or datetime.now(timezone.utc)
    llm_client = client or create_client(config)
    briefing_markdown = _build_briefing_payload(briefing, config)
    candidate_text = _build_candidate_payload(all_summaries)
    prompt = EVALUATION_USER_PROMPT_TEMPLATE.format(
        briefing_markdown=briefing_markdown,
        thread_summaries=candidate_text,
    )
    max_retries = config.summarizer.reduce.max_retries
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    for attempt in range(max_retries):
        try:
            response = _request_eval(llm_client, config, prompt)
        except Exception as exc:
            if attempt == max_retries - 1:
                LOGGER.error("Evaluation request failed after retries: %s", exc)
                return _empty_eval_result(briefing, all_summaries, config, generated_at)
            LOGGER.warning("Evaluation request attempt %s failed: %s", attempt + 1, exc)
            time.sleep(2**attempt)
            continue

        usage = response_token_usage(response)
        token_usage = {
            "input_tokens": token_usage["input_tokens"] + usage["input_tokens"],
            "output_tokens": token_usage["output_tokens"] + usage["output_tokens"],
        }
        raw_text = extract_response_text(response)
        try:
            payload = load_json_payload(raw_text)
            return _parse_eval_result(payload, briefing, all_summaries, response, generated_at, token_usage)
        except (ValueError, TypeError) as exc:
            if attempt == max_retries - 1:
                LOGGER.error("Evaluation JSON invalid after retries: %s", exc)
                return _empty_eval_result(briefing, all_summaries, config, generated_at)
            LOGGER.warning("Invalid evaluation JSON on attempt %s: %s", attempt + 1, exc)
            prompt = prompt + EVALUATION_JSON_RETRY_SUFFIX
            time.sleep(2**attempt)

    return _empty_eval_result(briefing, all_summaries, config, generated_at)


def save_eval_result(result: EvalResult, config: AppConfig) -> Path:
    """Persist evaluation result to output/eval/eval-YYYY-MM-DD.json."""

    eval_dir = config.root_dir / "output" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / f"eval-{result.date}.json"
    payload = {
        "date": result.date,
        "coverage": result.coverage,
        "diversity": result.diversity,
        "clarity": result.clarity,
        "redundancy": result.redundancy,
        "importance_calibration": result.importance_calibration,
        "notes": result.notes,
        "model": result.model,
        "generated_at": result.generated_at.astimezone(timezone.utc).isoformat(),
        "token_usage": result.token_usage,
        "candidate_count": result.candidate_count,
        "briefing_thread_count": result.briefing_thread_count,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _build_briefing_payload(briefing: FinalBriefing, config: AppConfig) -> str:
    """Render the briefing to a markdown string for the evaluation prompt."""

    from .formatter import render_briefing

    return render_briefing(briefing, config)


def _build_candidate_payload(summaries: list[ThreadSummary]) -> str:
    """Render all map-stage candidates as a compact text block."""

    return "\n".join(_candidate_block(summary) for summary in summaries)


def _candidate_block(summary: ThreadSummary) -> str:
    """Render one candidate summary for the evaluation prompt."""

    lines = [
        f"[{summary.thread_id}] {summary.topic} | importance={summary.importance}/10"
        f" | sources={summary.effective_source_count}",
        f"  {summary.headline_zh}",
        f"  {summary.summary_zh}",
        "---",
    ]
    return "\n".join(lines)


def _request_eval(client: Any, config: AppConfig, prompt: str) -> Any:
    """Send one JSON-mode evaluation request."""

    return client.chat.completions.create(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


def _parse_eval_result(
    payload: Any,
    briefing: FinalBriefing,
    all_summaries: list[ThreadSummary],
    response: Any,
    generated_at: datetime,
    token_usage: dict[str, int],
) -> EvalResult:
    """Convert the LLM JSON payload into an EvalResult."""

    if not isinstance(payload, dict):
        raise TypeError("Evaluation payload must be a JSON object")
    scores = {key: _coerce_score(payload.get(key)) for key in EVAL_SCORE_KEYS}
    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        notes = ""
    return EvalResult(
        date=briefing.date,
        coverage=scores["coverage"],
        diversity=scores["diversity"],
        clarity=scores["clarity"],
        redundancy=scores["redundancy"],
        importance_calibration=scores["importance_calibration"],
        notes=notes.strip(),
        model=str(getattr(response, "model", "")),
        generated_at=generated_at,
        token_usage=token_usage,
        candidate_count=len(all_summaries),
        briefing_thread_count=briefing.total_threads,
    )


def _empty_eval_result(
    briefing: FinalBriefing,
    all_summaries: list[ThreadSummary],
    config: AppConfig,
    generated_at: datetime,
) -> EvalResult:
    """Return a zero-scored result when evaluation fails entirely."""

    return EvalResult(
        date=briefing.date,
        coverage=0,
        diversity=0,
        clarity=0,
        redundancy=0,
        importance_calibration=0,
        notes="评估失败",
        model=config.llm.model,
        generated_at=generated_at,
        token_usage={"input_tokens": 0, "output_tokens": 0},
        candidate_count=len(all_summaries),
        briefing_thread_count=briefing.total_threads,
    )


def _coerce_score(value: Any) -> int:
    """Clamp a model score to the 1-10 range."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Score must be numeric, got {value!r}")
    return max(1, min(10, int(round(float(value)))))
