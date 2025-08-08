# strategies.py
import re

def score_and_tag(item: dict, cfg: dict) -> tuple[float, list[str]]:
    text = f"{item.get('title','')} {item.get('raw_text','')}".lower()
    cats = cfg.get("categories", {}) if isinstance(cfg, dict) else {}
    score = 0.0
    tags = []
    weights = {"macro":2.0,"rates":2.0,"earnings":1.5,"tech":1.4,"geopolitics":1.2,"energy":1.1,"fx":1.0,"crypto":0.8}

    for cat, kw_list in cats.items():
        for kw in kw_list:
            if kw.lower() in text:
                if cat not in tags: tags.append(cat)
                score += weights.get(cat, 1.0)

    tickers = set(re.findall(r'\b[A-Z]{2,6}\b', f"{item.get('title','')}"))
    score += min(len(tickers), 5) * 0.3

    src = (item.get("source") or "").lower()
    if any(k in src for k in ("reuters","bloomberg","wsj","ft")):
        score += 0.5

    return score, tags

def one_liner_en(item: dict, tags: list[str]) -> str:
    title = (item.get("title") or "").strip()
    title = re.sub(r'^\s*(breaking|update|exclusive)\s*:\s*', '', title, flags=re.I)
    if len(title) > 200: title = title[:197] + "..."
    return f"{title} [{' / '.join(tags[:3])}]" if tags else title
