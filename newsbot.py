#!/usr/bin/env python3
import asyncio, os, re, json, csv, sys, argparse, hashlib, time, io, math
from datetime import datetime, timedelta, timezone
import pytz
import httpx
from bs4 import BeautifulSoup
import feedparser
from dateutil import parser as dateparser
from aiocache import cached, SimpleMemoryCache

# Optional summarizer (sumy). If missing, we'll fall back.
try:
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    SUMY_OK = True
except Exception:
    SUMY_OK = False

# Optional pandas for CSV export across runs
try:
    import pandas as pd  # noqa: F401
    PANDAS_OK = True
except Exception:
    PANDAS_OK = False

# Optional Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

DEFAULT_TZ = os.getenv("NEWS_TZ", "America/Los_Angeles")

def now_tz():
    return datetime.now(pytz.timezone(DEFAULT_TZ))

def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = pytz.timezone(DEFAULT_TZ).localize(dt)
    return dt.astimezone(pytz.timezone(DEFAULT_TZ)).isoformat(timespec="seconds")

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def hash_key(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def safe_parse_date(s: str) -> datetime | None:
    if not s:
        return None
    try:
        dt = dateparser.parse(s)
        if dt is None: return None
        if dt.tzinfo is None:
            dt = pytz.timezone(DEFAULT_TZ).localize(dt)
        return dt.astimezone(pytz.timezone(DEFAULT_TZ))
    except Exception:
        return None

def summarize_text(text: str, sentences: int = 2) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""
    if SUMY_OK:
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summ = TextRankSummarizer()
            sents = [str(s) for s in summ(parser.document, sentences)]
            if sents:
                return " ".join(sents)
        except Exception:
            pass
    # fallback: truncate to 280 chars
    return (text[:277] + "...") if len(text) > 280 else text

@cached(ttl=3600, cache=SimpleMemoryCache)
async def fetch_text(client: httpx.AsyncClient, url: str, retries: int = 2) -> str:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = await client.get(url, timeout=20)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_exc = e
            try:
                status = getattr(r, 'status_code', 'n/a')
                snippet = (r.text or '')[:200]
            except Exception:
                status = 'n/a'; snippet = ''
            print(f"[FETCH_ERROR] attempt={attempt} {url} status={status} err={e}\n{snippet}\n")
            await asyncio.sleep(1.5 * (attempt + 1))
    raise last_exc

async def scrape_rss(client: httpx.AsyncClient, url: str, source_name: str, lookback_minutes: int | None = None):
    items = []
    text = await fetch_text(client, url)
    feed = feedparser.parse(text)
    cutoff = None
    if lookback_minutes:
        cutoff = now_tz() - timedelta(minutes=lookback_minutes)
    for e in feed.entries:
        title = normalize_whitespace(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        desc = BeautifulSoup(getattr(e, "summary", "") or getattr(e, "description", ""), "lxml").get_text(" ")
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        dt = safe_parse_date(published) or now_tz()
        if cutoff and dt < cutoff:
            continue
        items.append({
            "source": source_name,
            "title": title,
            "url": link,
            "published_at": to_iso(dt),
            "raw_text": normalize_whitespace(desc),
        })
    return items

async def scrape_html(client: httpx.AsyncClient, cfg: dict, source_name: str):
    url = cfg["url"]
    css = cfg.get("css", {})
    sel_item = css.get("item", "article")
    sel_title = css.get("title", "h2, h3, a")
    sel_time = css.get("time", "time")
    sel_body = css.get("body", "p")
    link_attr = css.get("link_attr", "href")

    html = await fetch_text(client, url)
    soup = BeautifulSoup(html, "lxml")
    out = []
    for node in soup.select(sel_item):
        title_node = node.select_one(sel_title)
        title = normalize_whitespace(title_node.get_text(" ")) if title_node else ""
        link = ""
        if title_node and title_node.has_attr(link_attr):
            link = title_node[link_attr]
        elif (a := node.select_one("a")) and a.has_attr("href"):
            link = a["href"]
        # Make absolute if relative
        link = httpx.URL(link, base=url).human_repr() if link else url

        time_node = node.select_one(sel_time)
        tstr = normalize_whitespace(time_node.get("datetime") or time_node.get_text(" ")) if time_node else ""
        dt = safe_parse_date(tstr) or now_tz()

        body_node = node.select_one(sel_body)
        body = normalize_whitespace(body_node.get_text(" ")) if body_node else ""

        if title:
            out.append({
                "source": source_name,
                "title": title,
                "url": link,
                "published_at": to_iso(dt),
                "raw_text": body
            })
    return out

async def scrape_playwright(cfg: dict, source_name: str):
    from playwright.async_api import async_playwright
    url = cfg["url"]
    wait_selector = cfg.get("wait_selector")
    item_selector = cfg.get("item_selector", "article")
    title_selector = cfg.get("title_selector", "h2, h3, .title")
    time_selector = cfg.get("time_selector", "time")
    link_selector = cfg.get("link_selector", "a")
    link_attr = cfg.get("link_attr", "href")

    out = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        if wait_selector:
            await page.wait_for_selector(wait_selector, timeout=15000)
        items = await page.query_selector_all(item_selector)
        for it in items[:100]:
            title_el = await it.query_selector(title_selector)
            title = normalize_whitespace(await title_el.inner_text()) if title_el else ""
            link_el = await it.query_selector(link_selector)
            link = await link_el.get_attribute(link_attr) if link_el else ""
            if link:
                link = httpx.URL(link, base=url).human_repr()
            time_el = await it.query_selector(time_selector)
            tstr = normalize_whitespace(await time_el.inner_text()) if time_el else ""
            dt = safe_parse_date(tstr) or now_tz()
            body_el = await it.query_selector("p, .body, .summary")
            body = normalize_whitespace(await body_el.inner_text()) if body_el else ""
            if title:
                out.append({
                    "source": source_name,
                    "title": title,
                    "url": link or url,
                    "published_at": to_iso(dt),
                    "raw_text": body
                })
        await browser.close()
    return out

def dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for it in items:
        key = hash_key((it.get("url") or "") + "|" + (it.get("title") or ""))
        if key in seen: 
            continue
        seen.add(key)
        out.append(it)
    return out


def _tokenize_upper(s: str) -> set[str]:
    s = s or ""
    # Split on non-alphanum, keep uppercase tokens for ticker matching
    toks = re.findall(r"[A-Z]{2,6}", s.upper())
    # Also capture $AAPL/$TSLA formats
    toks += [t[1:].upper() for t in re.findall(r"\$[A-Za-z]{1,6}", s)]
    return set(toks)

def filter_items(items: list[dict], cfg: dict, override_keywords=None, override_tickers=None) -> list[dict]:
    """Apply include filters. If include lists are non-empty, keep item if ANY matches."""
    filters = cfg.get("filters", {}) if isinstance(cfg, dict) else {}
    kw_list = [str(k).strip() for k in (override_keywords if override_keywords is not None else filters.get("include_keywords", [])) if str(k).strip()]
    tk_list = [str(k).strip().upper() for k in (override_tickers if override_tickers is not None else filters.get("include_tickers", [])) if str(k).strip()]
    if not kw_list and not tk_list:
        return items  # no filtering
    out = []
    kw_pat = re.compile("|".join([re.escape(k) for k in kw_list]), re.IGNORECASE) if kw_list else None
    for it in items:
        text = " ".join([it.get("title") or "", it.get("raw_text") or ""])
        hit_kw = bool(kw_pat.search(text)) if kw_pat else False
        toks = _tokenize_upper(text)
        hit_tk = any(t in toks for t in tk_list) if tk_list else False
        if hit_kw or hit_tk:
            out.append(it)
    return out

def _norm_url(u: str) -> str:
    try:
        url = httpx.URL(u)
        return httpx.URL(path=url.path, host=url.host, scheme=url.scheme).human_repr()
    except Exception:
        return u or ""

def dedupe_strong(items: list[dict]) -> list[dict]:
    """Dedupe by normalized URL and normalized title (case/whitespace)."""
    seen = set()
    out = []
    for it in items:
        url_key = _norm_url(it.get("url") or "")
        title_key = normalize_whitespace((it.get("title") or "").lower())
        key = hash_key(url_key + "|" + title_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def enrich_and_summarize(items: list[dict], max_sentences: int = 2) -> list[dict]:
    out = []
    for it in items:
        text = it.get("raw_text") or it.get("title") or ""
        summary = summarize_text(text, sentences=max_sentences)
        out.append({
            **it,
            "summary": summary
        })
    return out

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def run_once(cfg_path: str, send: bool = False, out_jsonl: str | None = "news.jsonl", out_csv: str | None = "news.csv", debug: bool = False, kw_override=None, tk_override=None):
    cfg = load_yaml(cfg_path)
    sources = [s for s in cfg.get("sources", []) if s.get("enabled", True)]
    all_items = []

    async with httpx.AsyncClient(headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36","Accept":"application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7", "Accept-Language":"en-US,en;q=0.9"}, timeout=20) as client:
        for s in sources:
            name = s["name"]
            typ = s["type"]
            try:
                if typ == "rss":
                    lookback = s.get("lookback_minutes")
                    items = await scrape_rss(client, s["url"], name, lookback)
                elif typ == "html":
                    items = await scrape_html(client, s, name)
                elif typ == "playwright":
                    items = await scrape_playwright(s, name)
                else:
                    print(f"[WARN] Unknown source type: {typ}", file=sys.stderr)
                    items = []
                print(f"[{name}] fetched {len(items)} items");
                if debug and items:
                    for _it in items[:3]:
                        print("   •", _it.get("title")[:120])
                all_items.extend(items)
            except Exception as e:
                print(f"[ERROR] {name}: {e}", file=sys.stderr)

    # Stronger dedupe
    all_items = dedupe_strong(all_items)
    # Apply filters (CLI overrides take precedence if provided)
    all_items = filter_items(all_items, cfg, override_keywords=kw_override, override_tickers=tk_override)
    # Summarize
    all_items = enrich_and_summarize(all_items)

    print(f"Total after filtering & dedupe: {len(all_items)} items")

    # Write outputs
    ts = to_iso(now_tz())
    if out_jsonl:
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for it in all_items:
                rec = {**it, "fetched_at": ts}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if out_csv:
        import os
        write_header = not os.path.exists(out_csv)
        import csv
        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fetched_at","source","published_at","title","summary","url"])
            if write_header:
                w.writeheader()
            for it in all_items:
                w.writerow({
                    "fetched_at": ts,
                    "source": it.get("source"),
                    "published_at": it.get("published_at"),
                    "title": it.get("title"),
                    "summary": it.get("summary"),
                    "url": it.get("url")
                })

    # Print to console: only per-source fetched lines are kept (above). No per-item output here.

    # Optional Telegram push
    if send and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID and all_items:
        await send_telegram_batch(all_items[:15])

async def send_telegram_batch(items: list[dict]):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    async with httpx.AsyncClient() as client:
        for it in items:
            text = f"📰 <b>{it['title']}</b>\n" \
                   f"📅 {it['published_at']}\n" \
                   f"🏷️ {it['source']}\n" \
                   f"{it['summary']}\n" \
                   f"{it['url']}"
            await client.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                             params={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"})
            await asyncio.sleep(0.3)

async def run_interval(cfg_path: str, seconds: int, send: bool):
    while True:
        try:
            await run_once(cfg_path, send)
        except Exception as e:
            print(f"[FATAL] run_once failed: {e}", file=sys.stderr)
        await asyncio.sleep(seconds)

def main():
    import yaml
    ap = argparse.ArgumentParser(description="Finance News Bot")
    ap.add_argument("--config", "-c", default="config.yaml")
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    ap.add_argument("--interval", type=int, default=0, help="Polling interval in seconds (0 = disabled)")
    ap.add_argument("--send", action="store_true", help="Send to Telegram if configured")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    ap.add_argument("--keywords", type=str, default="", help="Comma-separated keywords to include (case-insensitive)")
    ap.add_argument("--tickers", type=str, default="", help="Comma-separated tickers to include (e.g., NVDA,TSLA)")
    args = ap.parse_args()

    if args.once or args.interval <= 0:
        kw_ov = [s for s in (args.keywords.split(",") if args.keywords else []) if s.strip()]
        tk_ov = [s for s in (args.tickers.split(",") if args.tickers else []) if s.strip()]
        asyncio.run(run_once(args.config, send=args.send, debug=args.debug, kw_override=kw_ov if kw_ov else None, tk_override=tk_ov if tk_ov else None))
    else:
        asyncio.run(run_interval(args.config, args.interval, send=args.send))

if __name__ == "__main__":
    main()
