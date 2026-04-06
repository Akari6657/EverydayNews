# Daily Headline Agent

Daily Headline Agent collects RSS headlines from major news outlets, removes duplicates, summarizes the day in Chinese with an LLM, and delivers a daily Markdown briefing.

Daily Headline Agent（每日头条助手）会从多个国际新闻 RSS 源抓取头条，做跨媒体去重，再调用 LLM 生成中文摘要，并输出每日简报。

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
python -m src.main --dry-run
python -m src.main
```

## Configuration

- Edit `config.yaml` to add, remove, or reprioritize RSS sources.
- Fill `DEEPSEEK_API_KEY` in `.env`.
- Enable email or Telegram channels in `config.yaml` when needed.

## Scheduling

- Local scheduling: `python -m src.main --schedule`
- GitHub Actions: use `.github/workflows/daily-briefing.yml`
