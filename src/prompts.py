"""Prompt constants for the Daily Headline Agent summarization pipeline."""

from __future__ import annotations

MAP_SYSTEM_PROMPT = "你是一位资深国际新闻编辑，擅长从多家媒体报道中提炼同一事件的核心事实。你必须返回合法 JSON。"

MAP_USER_PROMPT_TEMPLATE = """你是一位资深国际新闻编辑。下面是若干条新闻事件簇，每一簇可能包含多个媒体对同一事件的报道。

对每一簇新闻，你需要输出一个 JSON 对象，包含以下字段：
- topic: 主题分类（从以下选一个：国际政治、经济金融、科技、社会民生、文化艺术、体育、科学健康、其他）
- headline_zh: 一句话中文标题（不超过 30 字）
- summary_zh: 2-3 句中文摘要，保留关键事实、数字、人物
- importance: 0-10 的重要性评分（10 = 头版头条级事件，0 = 无关紧要）
- entities: 关键实体列表（人名、地名、组织名），最多 5 个

重要性评分标准：
- 9-10: 重大国际事件、战争、重大政策变化、重要选举结果
- 7-8: 具有国际影响力的政治经济新闻、重大科技突破
- 5-6: 地区性重要新闻、值得关注的行业动态
- 3-4: 一般性报道
- 0-2: 娱乐八卦、地方琐事、标题党

返回格式要求：
- 只返回合法 JSON
- 顶层必须是一个对象，格式为 {{"items": [ ... ]}}
- items 中每个元素对应一个输入事件簇，顺序必须与输入一致
- 不要输出 Markdown、解释文字或代码块

事件簇列表：
{clusters_payload}
"""

MAP_JSON_RETRY_SUFFIX = """

重要提醒：你上一次没有返回可解析的 JSON。请这一次严格只返回一个 JSON 对象，格式必须是：
{"items": [{"topic": "...", "headline_zh": "...", "summary_zh": "...", "importance": 0, "entities": ["..."]}]}
不要输出任何其他文字。
"""

REDUCE_SYSTEM_PROMPT = "你是一位资深国际新闻主编，擅长将结构化新闻摘要编排为高质量中文日报。你必须返回合法 JSON。"

REDUCE_USER_PROMPT_TEMPLATE = """你是一位资深国际新闻主编。下面是今天各媒体头条新闻的结构化摘要列表。你的任务是生成一份高质量的中文日报。

请完成：
1. 写一段 3-5 句的今日新闻综述（overview_zh），概括今天最重要的事件和趋势
2. 将新闻按主题重新组织（可以合并相近主题、调整分类）
3. 每个主题下的新闻按重要性排序
4. 保留每条新闻的原始字段（cluster_id、headline_zh、summary_zh、source_names、primary_link、importance、entities）

返回严格的 JSON 对象，不要输出任何其他文字。
顶层格式必须是：
{{
  "overview_zh": "...",
  "topics": {{
    "国际政治": [
      {{
        "cluster_id": "...",
        "headline_zh": "...",
        "summary_zh": "...",
        "importance": 0,
        "entities": ["..."],
        "source_names": ["..."],
        "primary_link": "https://..."
      }}
    ]
  }}
}}

新闻摘要列表（已按重要性排序）：
{summaries_payload}
"""

REDUCE_JSON_RETRY_SUFFIX = """

重要提醒：你上一次没有返回可解析的 JSON。请这一次严格只返回一个 JSON 对象，格式必须包含：
{
  "overview_zh": "...",
  "topics": {
    "主题名": [
      {
        "cluster_id": "...",
        "headline_zh": "...",
        "summary_zh": "...",
        "importance": 0,
        "entities": ["..."],
        "source_names": ["..."],
        "primary_link": "https://..."
      }
    ]
  }
}
不要输出任何其他文字。
"""
