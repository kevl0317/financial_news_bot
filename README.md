# Finance News Bot (ET + Today-only) + Discord Batch Relay

- Fetches finance headlines (Yahoo Finance, Reuters via Google News, CNBC, Bloomberg, MarketWatch, Barron's, FT, SeekingAlpha, Nasdaq, Investing, WSJ).
- Robust timestamp parsing → converts to **ET** and keeps **today-only**.
- Writes JSONL/CSV; Discord relay every 30s.
- Discord messages are **clean text (no emojis)** and **batch-sent** so each update shows the bot's **name & icon once** (avoids stacked/misleading blocks).
- Link previews are suppressed by wrapping URLs in `<...>`.

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
playwright install           # (not used by default now; FJ/Jin10 disabled)
```

## Run the crawler
```bash
python newsbot.py --once --debug
python newsbot.py --interval 60
```

## Run the Discord relay (batch send)
```bash
python discord_bot.py --token YOUR_BOT_TOKEN --channel-id 123456789012345678 --interval 30 --tech --max 20 --verbose
# One-time backfill:
python discord_bot.py --channel-id 123456789012345678 --interval 30 --max 50 --reset-offset --verbose
```

## Config tips
- Filters in `config.yaml` are optional; you can override at runtime:
  - `--keywords earnings,rate`
  - `--tickers NVDA,TSLA`
  - `--tech` for a built-in tech preset (keywords + tickers)
