# 每日头条简报

从多家国际新闻 RSS 源自动抓取头条，将多家媒体对同一事件的报道按故事线聚类，调用 LLM 生成中文摘要，每天输出一份结构清晰的新闻简报。

## 功能

- **故事线聚类**：将不同媒体对同一事件的报道归为一组，而不是逐条堆砌
- **跨批次合并**：文章数量较多时分批聚类，批次间重复故事线自动合并
- **线内去重**：同一故事线内的近似重复报道自动过滤
- **两层简报**：按重要性分为「🔥 今日头条」和「📎 其他值得关注」
- **质量评估**：可选 LLM 评分，从覆盖度、多样性、清晰度等五个维度打分
- **多种输出与投递**：支持 Markdown、JSON，以及可选邮件、Telegram 推送

## 快速开始

**安装依赖**

```bash
pip install -r requirements.txt
```

**配置 API Key**

```bash
cp .env.example .env
# 在 .env 中填写 DEEPSEEK_API_KEY
# 标准运行、--dry-run、--dump-threads 都会用到 DeepSeek
```

**运行**

```bash
python -m src.main
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--config <文件>` | 使用指定配置文件，默认 `config.yaml` |
| `--dry-run` | 执行抓取、聚类、线内去重、排序，跳过 map/reduce 摘要生成与文件输出 |
| `--dump-threads` | 调用聚类流程并打印故事线分组结果，用于调试聚类效果 |
| `--dump-threads --dedup-within-threads` | 在打印前额外做一轮线内近重复去重 |
| `--eval` | 完整运行后额外调用 LLM 对简报质量打分 |
| `--schedule` | 按 `schedule.run_at` 配置的每日时间持续定时运行 |

## 配置

所有配置集中在 `config.yaml`。主要部分：

```yaml
# RSS 源列表
sources:
  - name: "BBC News"
    slug: "bbc"
    feeds:
      - url: "https://feeds.bbci.co.uk/news/rss.xml"
        category: "top"

# 抓取与筛选
pipeline:
  max_articles_per_source: 15
  importance_threshold: 5
  exclude_summary_keywords: ["最新动态", "持续更新", "live updates"]

# 线内近重复去重
dedup:
  method: "embedding"
  within_thread:
    similarity_threshold: 0.88

# 故事线聚类
thread_clustering:
  max_articles_per_call: 150   # 超过此数量时分批处理
  max_articles_per_thread: 12  # 单条故事线最多文章数
  enable_post_merge: true      # 启用启发式跨线合并
  enable_chunk_merge: true     # 启用分批后的 LLM 合并

# 排序与摘要
ranking:
  importance_floor: 0.15
  keep_major_always: true

summarizer:
  map:
    batch_size: 5
  reduce:
    top_k: 15

# 输出
output:
  markdown:
    directory: "output/md"
  json:
    directory: "output/json"

# 定时运行（每日 HH:MM）
schedule:
  timezone: "Asia/Shanghai"
  run_at: "08:00"
```

**可选环境变量**：

```bash
DEEPSEEK_API_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
SMTP_PASSWORD=
```

- 启用邮件推送时，需要在 `.env` 中提供 `SMTP_PASSWORD`
- 启用 Telegram 推送时，需要在 `.env` 中提供 `TELEGRAM_BOT_TOKEN` 和 `TELEGRAM_CHAT_ID`

**多配置文件**：如需按地区或主题生成不同简报，复制一份配置即可，代码不变：

```bash
cp config.yaml config.tech.yaml
python -m src.main --config config.tech.yaml
```

## 输出文件

| 路径 | 内容 |
|------|------|
| `output/md/YYYY-MM/briefing-YYYY-MM-DD.md` | Markdown 简报 |
| `output/json/YYYY-MM/briefing-YYYY-MM-DD.json` | 结构化 JSON |
| `output/eval/eval-YYYY-MM-DD.json` | 质量评分（需加 `--eval`） |

## 开发

```bash
# 运行全部测试
python -m pytest tests/

# 调试故事线聚类（会调用聚类 LLM，但不执行摘要阶段）
python -m src.main --dump-threads

# 预跑主流程（会抓取 / 聚类 / 线内去重 / 排序，但跳过成稿输出）
python -m src.main --dry-run
```

依赖 Python 3.12+，虚拟环境在 `.venv/`。如果没有激活虚拟环境，可用 `.venv/bin/python -m pytest tests/` 运行测试。
