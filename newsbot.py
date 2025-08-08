#!/usr/bin/env python3
import asyncio, os, re, json, csv, sys, argparse, hashlib, time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytz
import httpx
from bs4 import BeautifulSoup
import feedparser
from dateutil import parser as dateparser
from aiocache import cached, SimpleMemoryCache

# Optional summarizer (sumy). If missing, fallback to snippet
try:
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    SUMY_OK = True
except Exception:
    SUMY_OK = False

ET_TZ = ZoneInfo("America/New_York")
DEFAULT_TZ = os.getenv("NEWS_TZ", "America/Los_Angeles")  # used only for legacy fallbacks

def now_et():
    return datetime.now(ET_TZ)

def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET_TZ)
    return dt.astimezone(ET_TZ).isoformat(timespec="seconds")

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def hash_key(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def _normalize_datetime_str(s: str):
    """
    - Insert missing space between year and time: '20251:08 PM' -> '2025 1:08 PM'
    - Convert 'GMT+8'/'GMT-3' -> '+08:00'/'-03:00'
    - Strip trailing 'Updated X hours/minutes ago'
    """
    if not s:
        return "", None
    s0 = s.strip()

    rel_minutes = None
    m_rel_hr = re.search(r'Updated\s+(\d+)\s*hours?\s+ago', s0, flags=re.I)
    m_rel_min = re.search(r'Updated\s+(\d+)\s*minutes?\s+ago', s0, flags=re.I)
    if m_rel_hr:
        rel_minutes = int(m_rel_hr.group(1)) * 60
    elif m_rel_min:
        rel_minutes = int(m_rel_min.group(1))

    s1 = re.sub(r'Updated\s+\d+\s*(hours?|minutes?)\s+ago', '', s0, flags=re.I).strip()
    s1 = re.sub(r'(\b\d{4})(\d{1,2}:\d{2}\s*[AP]M\b)', r'\1 \2', s1, flags=re.I)

    def _gmt_to_offset(m):
        sign = m.group(1)
        hh = int(m.group(2))
        return f"{sign}{hh:02d}:00"
    s1 = re.sub(r'\bGMT\s*([+-])\s*(\d{1,2})\b', _gmt_to_offset, s1, flags=re.I)

    return s1.strip(), rel_minutes

def safe_parse_date(s: str) -> datetime | None:
    """
    Parse publisher strings robustly and return aware datetime in ET.
    Prefer absolute timestamps; if none but a relative 'Updated ... ago' exists, use that.
    """
    if not s:
        return None

    cleaned, rel_minutes = _normalize_datetime_str(s)

    tzinfos = {
        "ET": -5 * 3600, "EST": -5 * 3600, "EDT": -4 * 3600,
        "CT": -6 * 3600, "CST": -6 * 3600, "CDT": -5 * 3600,
        "MT": -7 * 3600, "MST": -7 * 3600, "MDT": -6 * 3600,
        "PT": -8 * 3600, "PST": -8 * 3600, "PDT": -7 * 3600,
        "GMT": 0,
    }

    dt_abs = None
    try:
        dt_abs = dateparser.parse(cleaned, tzinfos=tzinfos, fuzzy=True)
    except Exception:
        dt_abs = None

    if dt_abs:
        if dt_abs.tzinfo is None:
            dt_abs = dt_abs.replace(tzinfo=ET_TZ)
        return dt_abs.astimezone(ET_TZ)

    if rel_minutes is not None:
        return (now_et() - timedelta(minutes=rel_minutes)).astimezone(ET_TZ)

    return None

def is_today_et(dt: datetime) -> bool:
    try:
        today_et = now_et().date()
        return dt.astimezone(ET_TZ).date() == today_et
    except Exception:
        return False

@cached(ttl=3600, cache=SimpleMemoryCache)
async def fetch_text(client: httpx.AsyncClient, url: str, retries: int = 2) -> str:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = await client.get(url, timeout=30)
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
            await asyncio.sleep(1.0 * (attempt + 1))
    raise last_exc

async def scrape_rss(client: httpx.AsyncClient, url: str, source_name: str, lookback_minutes: int | None = None):
    items = []
    text = await fetch_text(client, url)
    feed = feedparser.parse(text)

    for e in feed.entries:
        title = normalize_whitespace(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        desc = BeautifulSoup(getattr(e, "summary", "") or getattr(e, "description", ""), "lxml").get_text(" ")
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        dt = safe_parse_date(published) or now_et()

        if not is_today_et(dt):
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
        link = httpx.URL(link, base=url).human_repr() if link else url

        time_node = node.select_one(sel_time)
        tstr = normalize_whitespace(time_node.get("datetime") or time_node.get_text(" ")) if time_node else ""
        dt = safe_parse_date(tstr) or now_et()

        if not is_today_et(dt):
            continue

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
    wait_selector = cfg.get("wait_selector") or "body"
    item_selector = cfg.get("item_selector", "article")
    title_selector = cfg.get("title_selector", "h2, h3, .title")
    time_selector = cfg.get("time_selector", "time, .time")
    link_selector = cfg.get("link_selector", "a")
    link_attr = cfg.get("link_attr", "href")

    out = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            await page.wait_for_selector(wait_selector, timeout=15000)
        except Exception:
            pass

        for _ in range(5):
            await page.mouse.wheel(0, 2500)
            await page.wait_for_timeout(1000)

        items = await page.query_selector_all(item_selector)
        for it in items[:200]:
            title_el = await it.query_selector(title_selector)
            title = (await title_el.inner_text()).strip() if title_el else ""

            link_el = await it.query_selector(link_selector)
            link = (await link_el.get_attribute(link_attr)) if link_el else ""
            if link:
                link = httpx.URL(link, base=url).human_repr()

            time_el = await it.query_selector(time_selector)
            tstr = (await time_el.inner_text()).strip() if time_el else ""
            dt = safe_parse_date(tstr) or now_et()

            if not is_today_et(dt):
                continue

            if title:
                out.append({
                    "source": source_name,
                    "title": normalize_whitespace(title),
                    "url": link or url,
                    "published_at": to_iso(dt),
                    "raw_text": ""
                })

        await browser.close()
    return out

def _tokenize_upper(s: str) -> set[str]:
    s = s or ""
    toks = re.findall(r"[A-Z]{2,6}", s.upper())
    toks += [t[1:].upper() for t in re.findall(r"\$[A-Za-z]{1,6}", s)]
    return set(toks)

def filter_items(items: list[dict], cfg: dict, override_keywords=None, override_tickers=None) -> list[dict]:
    filters = cfg.get("filters", {}) if isinstance(cfg, dict) else {}
    kw_list = [str(k).strip() for k in (override_keywords if override_keywords is not None else filters.get("include_keywords", [])) if str(k).strip()]
    tk_list = [str(k).strip().upper() for k in (override_tickers if override_tickers is not None else filters.get("include_tickers", [])) if str(k).strip()]
    if not kw_list and not tk_list:
        return items
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
        summary = ""  # title-only downstream; keep blank to minimize size
        out.append({**it, "summary": summary})
    return out

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

async def run_once(cfg_path: str, send: bool = False, out_jsonl: str | None = "news.jsonl", out_csv: str | None = "news.csv", debug: bool = False, kw_override=None, tk_override=None):
    cfg = load_yaml(cfg_path)
    sources = [s for s in cfg.get("sources", []) if s.get("enabled", True)]
    all_items = []

    async with httpx.AsyncClient(headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":"application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7",
        "Accept-Language":"en-US,en;q=0.9"
    }, timeout=30) as client:
        for s in sources:
            name = s["name"]
            typ = s["type"]
            try:
                if typ == "rss":
                    items = await scrape_rss(client, s["url"], name, s.get("lookback_minutes"))
                elif typ == "html":
                    items = await scrape_html(client, s, name)
                elif typ == "playwright":
                    items = await scrape_playwright(s, name)
                else:
                    print(f"[WARN] Unknown source type: {typ}", file=sys.stderr)
                    items = []
                print(f"[{name}] fetched {len(items)} items")
                if debug and items:
                    for _it in items[:3]:
                        print("   •", _it.get("title")[:120])
                all_items.extend(items)
            except Exception as e:
                print(f"[ERROR] {name}: {e}", file=sys.stderr)

    all_items = dedupe_strong(all_items)
    all_items = filter_items(all_items, cfg, override_keywords=kw_override, override_tickers=tk_override)
    all_items = enrich_and_summarize(all_items)
    print(f"Total after filtering & dedupe: {len(all_items)} items")

    ts = to_iso(now_et())
    if out_jsonl:
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for it in all_items:
                rec = {**it, "fetched_at": ts}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if out_csv:
        write_header = not os.path.exists(out_csv)
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

async def run_interval(cfg_path: str, seconds: int, send: bool, kw_override=None, tk_override=None):
    while True:
        try:
            await run_once(cfg_path, send, kw_override=kw_override, tk_override=tk_override)
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

    kw_ov = [s for s in (args.keywords.split(",") if args.keywords else []) if s.strip()]
    tk_ov = [s for s in (args.tickers.split(",") if args.tickers else []) if s.strip()]

    if args.once or args.interval <= 0:
        asyncio.run(run_once(args.config, send=args.send, debug=args.debug,
                             kw_override=kw_ov if kw_ov else None, tk_override=tk_ov if tk_ov else None))
    else:
        asyncio.run(run_interval(args.config, args.interval, send=args.send,
                                 kw_override=kw_ov if kw_ov else None, tk_override=tk_ov if tk_ov else None))

if __name__ == "__main__":
    main()
