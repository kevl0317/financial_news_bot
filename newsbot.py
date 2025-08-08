#!/usr/bin/env python3
import asyncio, os, re, json, csv, sys, argparse, hashlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import httpx
from bs4 import BeautifulSoup
import feedparser
from dateutil import parser as dateparser
from aiocache import cached, SimpleMemoryCache
from strategies import score_and_tag, one_liner_en

ET_TZ = ZoneInfo("America/New_York")

def now_et():
    return datetime.now(ET_TZ)

def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET_TZ)
    return dt.astimezone(ET_TZ).isoformat(timespec="seconds")

def normalize_whitespace(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", s or "").strip()

def hash_key(s: str) -> str:
    import hashlib
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def _normalize_datetime_str(s: str):
    import re as _re
    if not s:
        return "", None
    s0 = s.strip()
    rel_minutes = None
    m_rel_hr = _re.search(r'Updated\s+(\d+)\s*hours?\s+ago', s0, flags=_re.I)
    m_rel_min = _re.search(r'Updated\s+(\d+)\s*minutes?\s+ago', s0, flags=_re.I)
    if m_rel_hr:
        rel_minutes = int(m_rel_hr.group(1)) * 60
    elif m_rel_min:
        rel_minutes = int(m_rel_min.group(1))
    s1 = _re.sub(r'Updated\s+\d+\s*(hours?|minutes?)\s+ago', '', s0, flags=_re.I).strip()
    s1 = _re.sub(r'(\b\d{4})(\d{1,2}:\d{2}\s*[AP]M\b)', r'\1 \2', s1, flags=_re.I)
    def _gmt_to_offset(m):
        sign = m.group(1); hh = int(m.group(2))
        return f"{sign}{hh:02d}:00"
    s1 = _re.sub(r'\bGMT\s*([+-])\s*(\d{1,2})\b', _gmt_to_offset, s1, flags=_re.I)
    return s1.strip(), rel_minutes

def safe_parse_date(s: str):
    if not s:
        return None
    cleaned, rel_minutes = _normalize_datetime_str(s)
    tzinfos = {"ET":-5*3600,"EST":-5*3600,"EDT":-4*3600,"CT":-6*3600,"CST":-6*3600,"CDT":-5*3600,"MT":-7*3600,"MST":-7*3600,"MDT":-6*3600,"PT":-8*3600,"PST":-8*3600,"PDT":-7*3600,"GMT":0}
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
        return dt.astimezone(ET_TZ).date() == now_et().date()
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
                status = getattr(r, 'status_code', 'n/a'); snippet = (r.text or '')[:200]
            except Exception:
                status = 'n/a'; snippet = ''
            print(f"[FETCH_ERROR] attempt={attempt} {url} status={status} err={e}\n{snippet}\n")
            await asyncio.sleep(1.0 * (attempt + 1))
    raise last_exc

async def scrape_rss(client: httpx.AsyncClient, url: str, source_name: str):
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
        items.append({"source": source_name, "title": title, "url": link, "published_at": to_iso(dt), "raw_text": normalize_whitespace(desc)})
    return items

def _norm_url(u: str) -> str:
    try:
        url = httpx.URL(u)
        return httpx.URL(path=url.path, host=url.host, scheme=url.scheme).human_repr()
    except Exception:
        return u or ""

def dedupe_strong(items: list[dict]) -> list[dict]:
    seen = set(); out = []
    for it in items:
        url_key = _norm_url(it.get("url") or "")
        title_key = normalize_whitespace((it.get("title") or "").lower())
        key = hash_key(url_key + "|" + title_key)
        if key in seen: continue
        seen.add(key); out.append(it)
    return out

def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def postprocess(items: list[dict], cfg: dict) -> list[dict]:
    out = []
    for it in items:
        score, tags = score_and_tag(it, cfg)
        en = one_liner_en(it, tags)
        out.append({**it, "score": round(score,2), "tags": tags, "one_liner_en": en})
    return out

async def run_once(cfg_path: str, out_jsonl: str | None = "news.jsonl", out_csv: str | None = "news.csv", debug: bool = False):
    cfg = load_yaml(cfg_path)
    sources = [s for s in cfg.get("sources", []) if s.get("enabled", True)]
    all_items = []

    async with httpx.AsyncClient(headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept":"application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7",
        "Accept-Language":"en-US,en;q=0.9"
    }, timeout=30) as client:
        for s in sources:
            name = s["name"]; typ = s["type"]
            try:
                if typ == "rss":
                    items = await scrape_rss(client, s["url"], name)
                else:
                    items = []
                print(f"[{name}] fetched {len(items)} items")
                if debug and items:
                    for _it in items[:3]: print("   •", _it.get("title")[:120])
                all_items.extend(items)
            except Exception as e:
                print(f"[ERROR] {name}: {e}", file=sys.stderr)

    all_items = dedupe_strong(all_items)
    all_items = postprocess(all_items, cfg)
    print(f"Total after scoring & one-liners: {len(all_items)} items")

    ts = to_iso(now_et())
    if out_jsonl:
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for it in all_items:
                rec = {**it, "fetched_at": ts}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if out_csv:
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["fetched_at","source","published_at","title","one_liner_en","score","tags","url"])
            if write_header: w.writeheader()
            for it in all_items:
                w.writerow({"fetched_at": ts, "source": it.get("source"), "published_at": it.get("published_at"), "title": it.get("title"),
                            "one_liner_en": it.get("one_liner_en"), "score": it.get("score"),
                            "tags": "|".join(it.get("tags", [])), "url": it.get("url")})

async def run_interval(cfg_path: str, seconds: int):
    while True:
        try: await run_once(cfg_path)
        except Exception as e: print(f"[FATAL] run_once failed: {e}", file=sys.stderr)
        await asyncio.sleep(seconds)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Finance News Bot - No Translator")
    ap.add_argument("--config", "-c", default="config.yaml")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.once or args.interval <= 0:
        asyncio.run(run_once(args.config, debug=args.debug))
    else:
        asyncio.run(run_interval(args.config, args.interval))

if __name__ == "__main__":
    main()
