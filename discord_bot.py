#!/usr/bin/env python3
import os
import json
import sqlite3
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

import discord
from discord.ext import tasks

# === 关键：导入你已有的 newsbot.py 并直接调用其 run_once ===
import importlib.util

def import_newsbot(module_path: str):
    spec = importlib.util.spec_from_file_location("newsbot", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ---- 简单的“发送去重”存储（SQLite） ----
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

# ---- 读取 news.jsonl 新增行（用文件偏移避免重复读取）----
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
        # 更新偏移
        self.offset_state.write_text(str(size), encoding="utf-8")
        return new_items

# ---- Discord Bot ----
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
        self.loop_task = self._task_loop  # alias

    async def on_ready(self):
        print(f"Logged in as {self.user} (id={self.user.id})")  # type: ignore
        channel = self.get_channel(self.channel_id)
        if channel is None:
            print(f"[ERROR] Channel ID {self.channel_id} not found. Is the bot in the server?")
            return
        # 启动循环任务
        self._task_loop.change_interval(seconds=self.args.interval)
        self._task_loop.start()

    @tasks.loop(seconds=30.0)
    async def _task_loop(self):
        # 1) 调用 newsbot 进行一次抓取（复用你的 --keywords / --tickers 覆盖）
        kw_ov = [s for s in (self.args.keywords.split(",") if self.args.keywords else []) if s.strip()]
        tk_ov = [s for s in (self.args.tickers.split(",") if self.args.tickers else []) if s.strip()]
        try:
            await self.newsbot.run_once(
                self.args.config,
                send=False,
                debug=False,
                kw_override=kw_ov if kw_ov else None,
                tk_override=tk_ov if tk_ov else None
            )
        except Exception as e:
            print(f"[run_once ERROR] {e}")

        # 2) 读取本轮新增的 JSONL 行
        new_items = self.tail.read_new()
        if not new_items:
            print("[Discord] no new items.")
            return

        # 3) 去重 + 限量发送
        channel = self.get_channel(self.channel_id)
        if channel is None:
            print(f"[ERROR] Channel ID {self.channel_id} not found.")
            return

        sent_count = 0
        max_per_push = self.args.max
        for it in new_items:
            # 用 URL+title 做唯一键
            key = self.newsbot.hash_key((it.get("url") or "") + "|" + (it.get("title") or ""))
            if self.sent.has(key):
                continue
            # 发送到 Discord
            try:
                msg = self._format_item(it)
                await channel.send(msg)
                self.sent.add(key, it)
                sent_count += 1
                if sent_count >= max_per_push:
                    break
            except Exception as e:
                print(f"[Discord send ERROR] {e}")
                continue

        if sent_count:
            print(f"[Discord] pushed {sent_count} items.")

    from datetime import datetime

    def _format_item(self, it: dict) -> str:
        # Title only
        title = (it.get("title", "") or "").strip()

        # Format time: Month Day HH:MM (use local time from published_at ISO)
        date_str = it.get("published_at", "")
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            # Convert to local time if needed
            dt = dt.astimezone()
            date_fmt = dt.strftime("%b %d %H:%M")  # e.g., Aug 08 17:49
        except Exception:
            date_fmt = date_str

        src = it.get("source", "")
        url = it.get("url", "")

        # Suppress link preview if URL exists
        if url:
            url = f"<{url}>"

        return f"**{title}**\n📅 {date_fmt}  🏷️ {src}\n{url}"



def parse_args():
    ap = argparse.ArgumentParser(description="Discord relay for newsbot")
    ap.add_argument("--token", default=os.getenv("DISCORD_BOT_TOKEN",""), help="Discord bot token")
    ap.add_argument("--channel-id", required=True, help="Discord channel ID to send messages to")
    ap.add_argument("--newsbot-path", default=str(Path(__file__).parent / "newsbot.py"), help="Path to newsbot.py")
    ap.add_argument("--config", "-c", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--jsonl-path", default=str(Path(__file__).parent / "news.jsonl"))
    ap.add_argument("--state-dir", default=str(Path(__file__).parent / ".state"))
    ap.add_argument("--interval", type=int, default=30, help="Seconds between updates (default 30)")
    ap.add_argument("--max", type=int, default=5, help="Max items per push (default 5)")
    ap.add_argument("--keywords", type=str, default="", help="Comma-separated keywords")
    ap.add_argument("--tickers", type=str, default="", help="Comma-separated tickers, e.g. NVDA,TSLA")
    return ap.parse_args()

def main():
    args = parse_args()
    if not args.token:
        raise SystemExit("Missing Discord token. Set --token or DISCORD_BOT_TOKEN env.")
    intents = discord.Intents.default()
    client = NewsDiscordBot(intents=intents, args=args)
    client.run(args.token)

if __name__ == "__main__":
    main()
