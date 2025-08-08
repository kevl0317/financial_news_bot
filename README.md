# Finance News Bot (ET, Today-only) + Discord Relay

This project fetches finance headlines from multiple sources (Yahoo Finance, Reuters via Google News, CNBC, FinancialJuice, Jin10),
**parses timestamps robustly**, converts times to **Eastern Time (America/New_York)**, and **keeps only today's items in ET**.
It writes to JSONL/CSV and includes a Discord relay that posts **title + time (ET) + source** every 30 seconds with **no link preview**.
You can filter by keywords/tickers or a one-flag `--tech` preset.

> ⚠️ Respect site Terms of Service/robots.txt. Prefer RSS/APIs when possible.

## Quick Start
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
playwright install       # for FinancialJuice / Jin10
```

### Run crawler
```bash
# once
python newsbot.py --once --debug
# loop every 60s
python newsbot.py --interval 60 --debug
# keywords/tickers
python newsbot.py --once --keywords earnings,rate --tickers NVDA,TSLA
```

### Discord relay (every 30s)
```bash
# Windows CMD/PowerShell
python discord_bot.py --token YOUR_BOT_TOKEN --channel-id 123456789012345678 --interval 30 --tech --max 20 --verbose
# Or set env var in CMD:
#   set DISCORD_BOT_TOKEN=YOUR_BOT_TOKEN
# and run without --token
```

### One-time backfill of today's items
```bash
python discord_bot.py --channel-id 123456789012345678 --interval 30 --max 50 --reset-offset --verbose
```

## Files
- `newsbot.py` — crawler with **ET-aware date parsing** and **today-only in ET**
- `discord_bot.py` — Discord relay (title + `%b %d %H:%M` ET, no preview, `--tech` + filters, reset/verbose flags)
- `config.yaml` — sources & defaults
- `requirements.txt` — deps
- Outputs: `news.jsonl`, `news.csv`

