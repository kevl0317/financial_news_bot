# Finance News Bot (Polling + Summarization)

A lightweight Python bot that polls multiple finance news sources (e.g., Yahoo Finance, FinancialJuice, 金十) on a schedule,
summarizes each item, and saves the output with **date and source**. You can print to console, save to CSV/JSONL, and optionally
send to Telegram.

> ⚠️ **Respect robots.txt and site Terms of Service**. Some sites block scraping or require a license/API. Use RSS/official APIs
> whenever possible. The included scrapers are examples and may need adjustments.

## Features
- Async fetching with `httpx` (faster than sequential requests)
- Pluggable scrapers per source, configured via `config.yaml`
- Simple extractive summarization (TextRank via `sumy`) with graceful fallback
- Deduplication by URL/title (SQLite or in-memory)
- Output to console + JSONL/CSV with timestamp and source
- Optional Telegram push

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python newsbot.py --once        # run once
python newsbot.py --interval 60 # poll every 60 seconds
```

### Configure sources
Edit `config.yaml`. Examples included for:
- **Yahoo Finance** (RSS): `https://finance.yahoo.com/news/rssindex`
- **FinancialJuice** (homepage HTML): Example CSS selectors (may change).
- **金十 (Jin10)**: Uses Playwright (headless browser) to load dynamic content from https://www.jin10.com/flash .
  *Playwright is optional; disable if not needed.*

### Telegram (optional)
Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as env vars, then run:
```bash
python newsbot.py --interval 120 --send
```

## Notes
- If `sumy` isn't installed or fails, the bot will produce a short fallback snippet (first 280 chars).
- Playwright first run needs: `playwright install`.
- **Legal**: Always follow site ToS; prefer RSS/official APIs. This repo is for educational/demo purposes only.

