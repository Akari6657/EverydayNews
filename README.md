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

- `config.yaml` is the default **global headlines** profile. Keep it focused on mainstream international outlets and headline-quality filtering.
- If you want a different product flavor later, create a separate config file instead of overloading the default one.
  Examples: `config.north-america.yaml`, `config.tech.yaml`, `config.finance.yaml`
- All profiles run through the same pipeline code in `src.main`; only the config file changes.
- Edit `config.yaml` to add, remove, or reprioritize RSS sources.
- Fill `DEEPSEEK_API_KEY` in `.env`.
- Enable email or Telegram channels in `config.yaml` when needed.

## Config Profiles

Use `config.yaml` as the stable default briefing for "daily global headlines". When you need a regional or topic-specific edition, add a new config file and keep the default profile clean.

Examples:

```bash
python -m src.main --config config.yaml --dry-run
python -m src.main --config config.north-america.yaml --dry-run
python -m src.main --config config.tech.yaml
```

Recommended approach:

- Keep `config.yaml` focused on the main headline product.
- Put regional or topic-specific source mixes into separate `config.*.yaml` files.
- Reuse the same code, tests, and output pipeline for every profile.

## Scheduling

- Local scheduling: `python -m src.main --schedule`
