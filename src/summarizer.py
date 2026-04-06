"""Batch articles into a single DeepSeek summarization request."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from .models import AppConfig, Article, BriefingResult

LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = "你是一位资深国际新闻编辑，擅长将英文新闻翻译和摘要为高质量的中文简报。"
USER_PROMPT_TEMPLATE = """你是一位专业的国际新闻编辑。以下是今天来自多个国际主流媒体的头条新闻，请完成以下任务：

1. 将这些新闻按主题分类（如：国际政治、经济金融、科技、社会、体育等）
2. 合并报道同一事件的多条新闻
3. 为每条新闻写一个简洁的中文摘要（1-2句话），保留关键事实和数据
4. 标注每条新闻的来源媒体

输出格式要求：
- 使用 Markdown 格式
- 每个主题作为二级标题（##）
- 每条新闻包含：中文摘要、来源标注、原文链接
- 在文档开头用 3-5 句话概述今天的重大新闻趋势
- 语言风格：客观、简洁、信息密度高

以下是今天的新闻列表：

{articles_payload}
"""


def summarize_articles(
    articles: list[Article],
    config: AppConfig,
    client: Any | None = None,
    now: datetime | None = None,
) -> BriefingResult:
    """Summarize articles with a single batched LLM call."""

    generated_at = now or datetime.now(timezone.utc)
    if not articles:
        return _empty_briefing(config, generated_at)
    prompt = USER_PROMPT_TEMPLATE.format(articles_payload=_build_articles_payload(articles))
    llm_client = client or _create_client(config)
    response = _request_summary(llm_client, config, prompt)
    return _build_briefing_result(response, config, generated_at)


def _empty_briefing(config: AppConfig, generated_at: datetime) -> BriefingResult:
    """Return a fallback briefing when no fresh articles exist."""

    return BriefingResult(
        content="今日暂无新的头条新闻。",
        model=config.llm.model,
        token_usage={"input_tokens": 0, "output_tokens": 0},
        generated_at=generated_at,
    )


def _create_client(config: AppConfig) -> Any:
    """Create an OpenAI-compatible client for DeepSeek."""

    from openai import OpenAI

    api_key = os.getenv(config.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable '{config.llm.api_key_env}' is required")
    return OpenAI(api_key=api_key, base_url=config.llm.base_url)


def _request_summary(client: Any, config: AppConfig, prompt: str) -> Any:
    """Retry the LLM request up to three times."""

    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError("Failed to summarize articles after 3 attempts") from exc
            sleep_seconds = 2**attempt
            LOGGER.warning("Summarization attempt %s failed: %s", attempt + 1, exc)
            time.sleep(sleep_seconds)


def _build_articles_payload(articles: list[Article]) -> str:
    """Format articles into the prompt payload."""

    blocks = [_article_block(article) for article in articles]
    return "\n".join(blocks)


def _article_block(article: Article) -> str:
    """Render one article block for the model prompt."""

    published_text = article.published.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            "---",
            f"标题: {article.title}",
            f"来源: {article.source_name}",
            f"时间: {published_text}",
            f"摘要: {article.description}",
            f"链接: {article.link}",
            "---",
        ]
    )


def _build_briefing_result(response: Any, config: AppConfig, generated_at: datetime) -> BriefingResult:
    """Convert an SDK response into a briefing dataclass."""

    usage = getattr(response, "usage", None)
    token_usage = {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }
    return BriefingResult(
        content=_extract_response_text(response),
        model=str(getattr(response, "model", config.llm.model)),
        token_usage=token_usage,
        generated_at=generated_at,
    )


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
        return "\n".join(item.get("text", "") for item in content if isinstance(item, dict)).strip()
    return str(content).strip()
