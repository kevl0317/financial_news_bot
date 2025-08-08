#!/usr/bin/env python3
import os, json, sqlite3, asyncio, argparse
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import discord
from discord.ext import tasks
import importlib.util

ET_TZ = ZoneInfo("America/New_York")

TECH_KEYWORDS = [
    "tech","AI","artificial intelligence","chip","semiconductor","foundry","fab",
    "GPU","CPU","NPU","datacenter","cloud","SaaS","software","cybersecurity",
    "LLM","GPT","Llama","OpenAI","Anthropic","stability ai","AR","VR","XR",
    "quantum","robotics","autonomous","5G","edge","compute","server","hyperscaler",
    "TikTok","ByteDance","WeChat","Huawei","Apple","Google","Microsoft","Meta","Amazon"
]

TECH_TICKERS = [
    "NVDA","AMD","INTC","AVGO","ASML","ARM","MU","QCOM","TSM","SMCI",
    "AAPL","MSFT","GOOGL","GOOG","META","AMZN","TSLA","CRM","ADBE","ORCL",
    "NOW","PANW","CRWD","NET","ZS","SNOW","PLTR","SHOP","UBER","ABNB"
]

def import_newsbot(module_path: str):
    spec = importlib.util.spec_from_file_location("newsbot", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

class SentStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init()

    def _init(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sent (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    title TEXT,
                    url TEXT,
                    published_at TEXT,
                    sent_at TEXT
                )
            """)
            conn.commit()

    def has(self, item_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT 1 FROM sent WHERE id = ?", (item_id,))
            return cur.fetchone() is not None

    def add(self, item_id: str, it: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sent (id, source, title, url, published_at, sent_at) VALUES (?,?,?,?,?,?)",
                (
                    item_id,
                    it.get("source",""),
                    it.get("title",""),
                    it.get("url",""),
                    it.get("published_at",""),
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                )
            )
            conn.commit()

class JsonlTail:
    def __init__(self, path: Path, offset_state: Path):
        self.path = path
        self.offset_state = offset_state
        self.offset_state.parent.mkdir(parents=True, exist_ok=True)
        if not self.offset_state.exists():
            self.offset_state.write_text("0", encoding="utf-8")

    def read_new(self):
        new_items = []
        last = int(self.offset_state.read_text().strip() or "0")
        if not self.path.exists():
            return new_items
        size = self.path.stat().st_size
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(last)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    new_items.append(obj)
                except Exception:
                    continue
        self.offset_state.write_text(str(size), encoding="utf-8")
        return new_items

class NewsDiscordBot(discord.Client):
    def __init__(self, *, intents: discord.Intents, args):
        super().__init__(intents=intents)
        self.args = args
        self.channel_id = int(args.channel_id)
        self.newsbot = import_newsbot(args.newsbot_path)
        self.sent = SentStore(Path(args.state_dir) / "sent.db")
        self.tail = JsonlTail(
            path=Path(args.jsonl_path),
            offset_state=Path(args.state_dir) / "jsonl.offset"
        )

    async def on_ready(self):
        print(f"Logged in as {self.user} (id={self.user.id})")  # type: ignore
        channel = self.get_channel(self.channel_id)
        if channel is None:
            print(f"[ERROR] Channel ID {self.channel_id} not found. Is the bot in the server?")
            return

        if self.args.reset_offset:
            try:
                (Path(self.args.state_dir) / "jsonl.offset").write_text("0", encoding="utf-8")
                import sqlite3
                with sqlite3.connect(Path(self.args.state_dir) / "sent.db") as conn:
                    conn.execute("DELETE FROM sent")
                    conn.commit()
                print("[State] Reset offset and cleared sent cache.")
            except Exception as e:
                print(f"[State] Reset failed: {e}")

        self._task_loop.change_interval(seconds=self.args.interval)
        self._task_loop.start()

    def _parse_iso_to_et(self, s: str):
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                from zoneinfo import ZoneInfo
                dt = dt.replace(tzinfo=ZoneInfo("UTC"))
            return dt.astimezone(ET_TZ)
        except Exception:
            return None

    @tasks.loop(seconds=30.0)
    async def _task_loop(self):
        kw_ov = [s for s in (self.args.keywords.split(",") if self.args.keywords else []) if s.strip()]
        tk_ov = [s for s in (self.args.tickers.split(",") if self.args.tickers else []) if s.strip()]
        if self.args.tech:
            kw_ov = sorted(set((kw_ov or []) + TECH_KEYWORDS))
            tk_ov = sorted(set((tk_ov or []) + TECH_TICKERS))

        try:
            await self.newsbot.run_once(
                self.args.config, send=False, debug=self.args.verbose,
                kw_override=kw_ov if kw_ov else None,
                tk_override=tk_ov if tk_ov else None
            )
        except Exception as e:
            print(f"[run_once ERROR] {e}")

        new_items = self.tail.read_new()
        if self.args.verbose:
            print(f"[Discord] tail new lines: {len(new_items)}")
        if not new_items:
            return

        # Only today (ET) and sort
        today_et = datetime.now(ET_TZ).date()
        today_items = []
        for it in new_items:
            dt_et = self._parse_iso_to_et(it.get("published_at",""))
            if dt_et and dt_et.date() == today_et:
                it["_dt_et"] = dt_et
                today_items.append(it)
        today_items.sort(key=lambda x: x["_dt_et"])

        if self.args.verbose:
            print(f"[Discord] today-only after tail: {len(today_items)}")
        if not today_items:
            return

        channel = self.get_channel(self.channel_id)
        if channel is None:
            print(f"[ERROR] Channel ID {self.channel_id} not found.")
            return

        # Build clean messages (no emojis), batch send
        msgs = []
        for it in today_items:
            key = self.newsbot.hash_key((it.get("url") or "") + "|" + (it.get("title") or ""))
            if self.sent.has(key):
                continue
            src = it.get("source","")
            title = (it.get("title","") or "").strip()
            url = it.get("url","")
            dt_et = it.get("_dt_et") or self._parse_iso_to_et(it.get("published_at",""))
            time_str = dt_et.strftime("%b %d %H:%M ET") if dt_et else (it.get("published_at","") or "")
            line = f"***{title}***\nTime: {time_str}"
            # if url:
            #     line += f"\n<{url}>"  # suppress preview
            msgs.append(line)
            self.sent.add(key, it)
            if len(msgs) >= self.args.max:
                break

        if msgs:
            combined = "\n\n".join(msgs)
            try:
                await channel.send(combined)
            except Exception as e:
                print(f"[Discord send ERROR] {e}")

    def _format_item(self, it: dict) -> str:
        # Not used in batch mode; kept for compatibility
        src = it.get("source","")
        title = (it.get("title","") or "").strip()
        dt_et = it.get("_dt_et") or self._parse_iso_to_et(it.get("published_at",""))
        time_str = dt_et.strftime("%b %d %H:%M ET") if dt_et else (it.get("published_at","") or "")
        url = it.get("url","")
        line = f"[{src}] {title}\nTime: {time_str}"
        # if url: line += f"\n<{url}>"
        return line

def parse_args():
    ap = argparse.ArgumentParser(description="Discord relay for newsbot")
    ap.add_argument("--token", default=os.getenv("DISCORD_BOT_TOKEN",""), help="Discord bot token")
    ap.add_argument("--channel-id", required=True, help="Discord channel ID to send messages to")
    ap.add_argument("--newsbot-path", default=str(Path(__file__).parent / "newsbot.py"), help="Path to newsbot.py")
    ap.add_argument("--config", "-c", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--jsonl-path", default=str(Path(__file__).parent / "news.jsonl"))
    ap.add_argument("--state-dir", default=str(Path(__file__).parent / ".state"))
    ap.add_argument("--interval", type=int, default=30, help="Seconds between updates (default 30)")
    ap.add_argument("--max", type=int, default=5, help="Max items per batch (default 5)")
    ap.add_argument("--keywords", type=str, default="", help="Comma-separated keywords")
    ap.add_argument("--tickers", type=str, default="", help="Comma-separated tickers, e.g. NVDA,TSLA")
    ap.add_argument("--tech", action="store_true", help="Use built-in tech filters (keywords+tickers)")
    ap.add_argument("--reset-offset", action="store_true", help="Reset JSONL offset and clear sent cache on startup.")
    ap.add_argument("--verbose", action="store_true", help="Extra logs: tail counts and today counts.")

    return ap.parse_args()

def main():
    args = parse_args()
    token = args.token or os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit("Missing Discord token. Set --token or DISCORD_BOT_TOKEN.")

    intents = discord.Intents.default()
    client = NewsDiscordBot(intents=intents, args=args)
    try:
        client.run(token, reconnect=True)
    except discord.errors.LoginFailure:
        raise SystemExit("LoginFailure: invalid token (double-check).")

if __name__ == "__main__":
    main()
