"""Prompt constants for the Daily Headline Agent summarization pipeline."""

from __future__ import annotations

THREAD_CLUSTERING_SYSTEM_PROMPT = "你是一位资深国际新闻编辑，擅长从多家媒体标题中识别同一故事线。你必须返回合法 JSON。"

THREAD_CLUSTERING_PROMPT_TEMPLATE = """你是一位资深国际新闻编辑。下面是今天从多家国际媒体抓取到的头条新闻列表。
请将它们按“故事线（story thread）”分组。

故事线的定义：报道同一个正在发展的事件、议题或话题的文章应归入同一组，即使切入角度不同。

分组原则：
1. 宁可粗分不要细分。如果多条新闻属于同一个大事件的不同侧面，合并到同一故事线。
2. 但不要把一个超级事件的所有“后果”和“外围反应”都塞进一个巨大的故事线。
   如果已经明显形成不同编辑角度，请拆成更紧凑的子线，例如：
   - 停火协议本身 / 条款与谈判
   - 停火后的直接军事行动或地区外溢
   - 油价、航运、市场、供应链等经济影响
   - 某国国内政治反应或舆论反应
3. 独立新闻各自成组，不要为了凑数而强行合并。
4. 像 “Morning news brief”“Watch:” 这类通用包装标题，如果本身信息量很弱，通常单独成组，不要用它们把不相干文章拉到一起。
5. 不能仅因为“都属于政治新闻 / 都来自同一媒体 / 都发生在同一地区”就合并。必须有明确的共同事件锚点。
6. 如果某一组中的文章标题共享的人名、地点、协议、行动、市场反应并不明显一致，宁可拆开。
7. 所有文章必须被分配到某个故事线，不能遗漏。
8. 返回严格 JSON，不要输出任何解释、Markdown 或代码块。

返回格式必须是：
{{
  "threads": [
    {{
      "thread_id": 1,
      "topic": "简短中文主题",
      "topic_en": "Short English topic",
      "article_ids": [1, 2, 5],
      "rationale": "一句话说明这些文章为什么属于同一故事线"
    }}
  ]
}}

文章列表：
{articles_payload}
"""

THREAD_REFINEMENT_PROMPT_TEMPLATE = """你是一位资深国际新闻编辑。下面是一组被初步归为同一故事线的文章，但这组可能过于宽泛。
请把它们重新拆成 2-6 个更紧凑的子故事线。

拆分原则：
1. 保持“同一核心叙事”在同一组内。
2. 如果已经明显出现不同报道角度，应拆开，例如：
   - 协议/谈判本身
   - 军事行动与地区安全后续
   - 市场、能源、供应链等经济冲击
   - 国内政治/选举/舆论反应
3. 不要因为都和同一个大事件有关，就把所有文章塞进一个组。
4. 通用包装标题（如 Watch、Morning news brief）如果信息量弱，可以单独成组。
5. 所有文章必须被分配，不能遗漏。

返回严格 JSON，格式必须是：
{{
  "threads": [
    {{
      "thread_id": 1,
      "topic": "...",
      "topic_en": "...",
      "article_ids": [1, 2],
      "rationale": "..."
    }}
  ]
}}

当前大故事线：
topic: {topic}
topic_en: {topic_en}
articles:
{articles_payload}
"""

THREAD_CLUSTERING_JSON_RETRY_SUFFIX = """

重要提醒：你上一次没有返回可解析的 JSON。请这一次严格只返回一个 JSON 对象，格式必须是：
{
  "threads": [
    {
      "thread_id": 1,
      "topic": "...",
      "topic_en": "...",
      "article_ids": [1, 2],
      "rationale": "..."
    }
  ]
}
不要输出任何其他文字。
"""

THREAD_MAP_SYSTEM_PROMPT = "你是一位资深国际新闻编辑，擅长将同一故事线下的多源报道整合成结构化中文摘要。你必须返回合法 JSON。"

THREAD_MAP_USER_PROMPT_TEMPLATE = """你是一位资深国际新闻编辑。下面是若干个“故事线”，每个故事线包含多家媒体对同一持续事件或议题的不同报道。

对每个故事线，你需要输出一个 JSON 对象，包含以下字段：
- topic: 主题分类（从以下选一个：国际政治、经济金融、科技、社会民生、文化艺术、体育、科学健康、其他）
- headline_zh: 一句话中文标题（不超过 30 字）
- summary_zh: 2-4 句中文摘要，整合该故事线中的关键事实、进展、分歧与背景
- importance: 0-10 的重要性评分（10 = 头版头条级事件，0 = 无关紧要）
- entities: 关键实体列表（人名、地名、组织名），最多 5 个

重要提示：
1. 同一故事线里的文章可能来自不同媒体、不同角度；请合并成一个综合摘要，不要逐条复述。
2. 如果故事线里存在多个角度，请优先突出最重要的 1-2 个角度，而不是面面俱到。
3. 不要因为故事线标题里写了某个主题，就机械照搬；请根据文章内容判断主题分类。
4. 返回格式要求：
   - 只返回合法 JSON
   - 顶层必须是一个对象，格式为 {{"items": [ ... ]}}
   - items 中每个元素对应一个输入故事线，顺序必须与输入一致
   - 不要输出 Markdown、解释文字或代码块

故事线列表：
{threads_payload}
"""

THREAD_MAP_JSON_RETRY_SUFFIX = """

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
4. 每个主题下只返回 thread_id 列表，顺序就是最终展示顺序
5. 不要重复输出 headline_zh、summary_zh、source_names、primary_link、importance、entities，这些字段会由程序根据 thread_id 自动补全

返回严格的 JSON 对象，不要输出任何其他文字。
顶层格式必须是：
{{
  "overview_zh": "...",
  "topics": {{
    "国际政治": ["thread_id_1", "thread_id_2"],
    "经济金融": ["thread_id_3"]
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
    "主题名": ["thread_id_1", "thread_id_2"]
  }
}
不要输出任何其他文字。
"""

REDUCE_SAFE_USER_PROMPT_TEMPLATE = """你是一位资深国际新闻主编。下面是今天候选新闻的精简列表。请避免复述候选中的敏感细节，只基于主题、简短标题和重要性做高层概括。

请完成：
1. 写一段 2-4 句的今日新闻综述（overview_zh），用中性、概括性的中文表达
2. 将新闻按主题组织
3. 每个主题下只返回 thread_id 列表，顺序就是最终展示顺序
4. 不要输出任何其他文字

返回严格的 JSON 对象，格式必须是：
{{
  "overview_zh": "...",
  "topics": {{
    "国际政治": ["thread_id_1", "thread_id_2"],
    "经济金融": ["thread_id_3"]
  }}
}}

候选新闻列表：
{summaries_payload}
"""

EVALUATION_SYSTEM_PROMPT = "你是一位严格的新闻质量评估员，擅长根据候选新闻和最终简报评估质量。你必须返回合法 JSON。"

EVALUATION_USER_PROMPT_TEMPLATE = """你是一位新闻质量评估员。请对下面这份每日新闻简报进行评分（每项 1-10）：

1. coverage: 是否覆盖了今天的重大事件（基于候选新闻列表对照）
2. diversity: 主题多样性（是否过于集中某一领域）
3. clarity: 中文摘要的清晰度和可读性
4. redundancy: 条目间是否有重复（10 = 无重复，1 = 大量重复）
5. importance_calibration: 重要性评分是否合理

返回严格的 JSON 对象，不要输出任何其他文字。格式必须是：
{{
  "coverage": 0,
  "diversity": 0,
  "clarity": 0,
  "redundancy": 0,
  "importance_calibration": 0,
  "notes": "简短评语"
}}

简报内容：
{briefing_markdown}

候选新闻列表（用于对照）：
{thread_summaries}
"""

EVALUATION_JSON_RETRY_SUFFIX = """

重要提醒：你上一次没有返回可解析的 JSON。请这一次严格只返回一个 JSON 对象，格式必须包含：
{
  "coverage": 0,
  "diversity": 0,
  "clarity": 0,
  "redundancy": 0,
  "importance_calibration": 0,
  "notes": "简短评语"
}
不要输出任何其他文字。
"""
