# 每日头条简报

从多家国际新闻 RSS 源自动抓取头条，将多媒体报道按故事线聚类，调用 LLM 生成中文摘要，每天输出一份结构清晰的新闻简报。

## 功能

- **故事线聚类**：将不同媒体对同一事件的报道归为一组，而不是逐条堆砌
- **跨批次合并**：文章数量较多时分批聚类，批次间重复故事线自动合并
- **线内去重**：同一故事线内的近似重复报道自动过滤
- **两层简报**：按重要性分为「🔥 今日头条」和「📎 其他值得关注」
- **质量评估**：可选 LLM 评分，从覆盖度、多样性、清晰度等五个维度打分
- **多渠道推送**：支持 Markdown 文件、JSON、邮件、Telegram

## 快速开始

**安装依赖**

```bash
pip install -r requirements.txt
```

**配置 API Key**

```bash
cp .env.example .env
# 在 .env 中填写 DEEPSEEK_API_KEY
```

**运行**

```bash
python -m src.main
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--config <文件>` | 使用指定配置文件，默认 `config.yaml` |
| `--dry-run` | 只执行抓取、聚类、排序，不调用 LLM 生成摘要 |
| `--dump-threads` | 打印故事线分组结果后退出，用于调试聚类效果 |
| `--dump-threads --dedup-within-threads` | 在打印前对线内文章做近似去重 |
| `--eval` | 完整运行后额外调用 LLM 对简报质量打分 |
| `--schedule` | 按 `config.yaml` 中的 cron 配置持续定时运行 |

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

# 故事线聚类
thread_clustering:
  max_articles_per_call: 150   # 超过此数量时分批处理
  max_articles_per_thread: 12  # 单条故事线最多文章数
  enable_post_merge: true      # 启用启发式跨线合并
  enable_chunk_merge: true     # 启用分批后 LLM 合并 pass

# 定时运行
schedule:
  timezone: "Asia/Shanghai"
  run_at: "08:00"
```

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
| `output/metrics.jsonl` | 每次运行的指标（文章数、token 用量、耗时） |

## 开发

```bash
# 运行全部测试
pytest tests/

# 调试聚类效果（不消耗 LLM token）
python -m src.main --dump-threads
```

依赖 Python 3.12+，虚拟环境在 `.venv/`。
