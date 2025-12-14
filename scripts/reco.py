import os
import re
import json
import math
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# 我只关心gr-qc和astro-ph的论文，如果想增加来源，比如hep-th的论文，就在这个地方添加
LIST_PAGES = {
    "gr-qc": "https://arxiv.org/list/gr-qc/new",
    "astro-ph": "https://arxiv.org/list/astro-ph/new",
}

EMBED_MODEL = "text-embedding-3-small"   # 便宜且足够好
TOP_N = 40                               # 页面展示/归档的 Top-N
MAX_CHARS_PER_PAPER = 6000               # 避免过长文本（embedding 有 token 上限）:contentReference[oaicite:2]{index=2}

HEADERS = {
    "User-Agent": "arxiv-reco (GitHub Actions); contact: your-email@example.com"
}

def ensure_dirs():
    os.makedirs("docs/data", exist_ok=True)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<","&lt;").replace(">","&gt;")

def parse_listing_date(soup: BeautifulSoup) -> str:
    # e.g. "Showing new listings for Friday, 12 December 2025"
    h3 = soup.find("h3", string=re.compile(r"Showing new listings for", re.I))
    if not h3:
        # fallback：有时 h3 不是纯文本，改用 get_text 扫描
        for h in soup.find_all("h3"):
            t = h.get_text(" ", strip=True)
            if t.lower().startswith("showing new listings for"):
                h3 = h
                break
    if not h3:
        raise RuntimeError("Cannot find listing date header on /new page.")

    text = h3.get_text(" ", strip=True)
    m = re.search(r"Showing new listings for\s+(.*)$", text, re.I)
    if not m:
        raise RuntimeError(f"Cannot parse listing date from header: {text}")

    dt = datetime.strptime(m.group(1).strip(), "%A, %d %B %Y")
    return dt.date().isoformat()

def parse_all_entries(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    out = []
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt", recursive=False)
        dds = dl.find_all("dd", recursive=False)
        for dt, dd in zip(dts, dds):
            dt_text = dt.get_text(" ", strip=True)
            if "(replaced)" in dt_text:
                continue

            abs_a = dt.find("a", href=re.compile(r"^/abs/"))
            if not abs_a or not abs_a.get("href"):
                continue

            link = "https://arxiv.org" + abs_a["href"].strip()
            arxiv_id = abs_a.get_text(strip=True).replace("arXiv:", "").strip()

            title_div = dd.find("div", class_=re.compile(r"list-title"))
            title = title_div.get_text(" ", strip=True) if title_div else ""
            title = re.sub(r"^\s*Title:\s*", "", title).strip()

            auth_div = dd.find("div", class_=re.compile(r"list-authors"))
            authors = [a.get_text(" ", strip=True) for a in auth_div.find_all("a")] if auth_div else []

            fulltext = dd.get_text(" ", strip=True)

            out.append({
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": authors,
                "link": link,
                "fulltext": fulltext,
            })
    return out

def truncate(s: str, max_chars: int) -> str:
    s = s or ""
    return s if len(s) <= max_chars else s[:max_chars] + " …"

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    # 批处理，避免一次请求过大
    vecs = []
    BATCH = 96
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        resp = client.embeddings.create(model=model, input=chunk)
        # 保持顺序
        vecs.extend([d.embedding for d in resp.data])
    return vecs

def render_html(latest_day: str, ranked_items: List[Dict[str, Any]], sources_text: str) -> str:
    if not ranked_items:
        items_html = "<p>该批次无推荐条目。</p>"
    else:
        blocks = []
        for it in ranked_items:
            authors_list = it.get("authors", []) or []
            authors = ", ".join(authors_list)
            authors_for_data = "|".join(authors_list)
            blocks.append(
                f"<div class='item' data-authors='{esc(authors_for_data)}'>"
                f"  <div class='idline'><b>{esc(it.get('arxiv_id',''))}</b> <span class='score'>score={it.get('score',0):.3f}</span> <span class='src'>{esc(it.get('source',''))}</span></div>"
                f"  <div class='title'><a href='{esc(it.get('link',''))}' target='_blank' rel='noopener'>{esc(it.get('title',''))}</a></div>"
                f"  <div class='authors authors-hidden'>{esc(authors)}</div>"
                "</div>"
            )
        items_html = "\n".join(blocks)

    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>arXiv Recommender ({latest_day})</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial; margin: 24px; line-height: 1.5; }}
    .meta {{ color: #555; margin-bottom: 16px; }}
    .item {{ padding: 12px 0; border-bottom: 1px solid #eee; }}
    .idline {{ font-size: 14px; }}
    .score {{ color:#666; margin-left: 10px; }}
    .src {{ color:#666; margin-left: 10px; }}
    .title {{ margin: 6px 0; }}
    .authors-hidden {{ display: none; }}
    button {{ padding: 6px 10px; border: 1px solid #ccc; background: #fff; border-radius: 8px; cursor: pointer; }}
    code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>arXiv Recommender ({latest_day})</h1>
  <div class="meta">
    来源：{sources_text}；方法：Embeddings 相似度（cosine）对 <code>profile.md</code> 与论文文本匹配并排序。<br/>
    输出 Top {TOP_N}。
  </div>

  <div style="margin: 10px 0 18px 0;">
    <button id="toggleAuthors">显示/隐藏作者</button>
  </div>

  <h2>本批次推荐</h2>
  {items_html}

<script>
  document.getElementById("toggleAuthors").addEventListener("click", () => {{
    document.querySelectorAll(".authors").forEach(el => el.classList.toggle("authors-hidden"));
  }});
</script>
</body>
</html>
"""

def main():
    ensure_dirs()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var. Add it as a GitHub Actions secret and pass it to the workflow.")

    client = OpenAI(api_key=api_key)

    profile = load_text("profile.md").strip()
    if not profile:
        raise RuntimeError("profile.md is empty.")

    parsed = {}
    dates = []

    for name, url in LIST_PAGES.items():
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        day = parse_listing_date(soup)
        dates.append(day)

        items = parse_all_entries(soup)
        for it in items:
            it["source"] = name
        parsed[name] = (day, items)

    latest_day = max(dates)

    # 合并同一批次的候选论文
    candidates = []
    for _, (day, items) in parsed.items():
        if day != latest_day:
            continue
        candidates.extend(items)
    # 去重：同一 arXiv_id 可能同时出现在多个 /new 页面
    by_id = {}
    for it in candidates:
        k = it.get("arxiv_id", "")
        if not k:
            continue
        if k not in by_id:
            by_id[k] = dict(it)
            by_id[k]["_sourceset"] = {it.get("source", "")} if it.get("source") else set()
        else:
            if it.get("source"):
                by_id[k]["_sourceset"].add(it["source"])
            # 如果不同来源抓到的 fulltext 长度不同，保留更长的（更有信息量）
            if len(it.get("fulltext", "")) > len(by_id[k].get("fulltext", "")):
                keep_set = by_id[k]["_sourceset"]
                by_id[k] = dict(it)
                by_id[k]["_sourceset"] = keep_set

    candidates = []
    for _, it in by_id.items():
        sources = sorted(list(it.pop("_sourceset", set())))
        it["source"] = ", ".join([s for s in sources if s])
        candidates.append(it)

    # 为 embedding 组装文本（尽量包含摘要段落信息）
    paper_texts = []
    for it in candidates:
        t = f"Title: {it.get('title','')}\nAuthors: {', '.join(it.get('authors',[]))}\nText: {it.get('fulltext','')}"
        paper_texts.append(truncate(t, MAX_CHARS_PER_PAPER))

    # 计算向量
    profile_vec = np.array(embed_texts(client, [profile], EMBED_MODEL)[0], dtype=np.float32)
    paper_vecs = embed_texts(client, paper_texts, EMBED_MODEL)

    ranked = []
    for it, v in zip(candidates, paper_vecs):
        score = cosine(profile_vec, np.array(v, dtype=np.float32))
        ranked.append({
            "source": it.get("source",""),
            "arxiv_id": it.get("arxiv_id",""),
            "title": it.get("title",""),
            "authors": it.get("authors",[]) or [],
            "link": it.get("link",""),
            "score": score,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked[:TOP_N]

    # 输出 JSON 与 HTML
    day_path = f"docs/data/{latest_day}.json"
    with open(day_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)

    sources_text = " 与 ".join([f"<code>{k}/new</code>" for k in LIST_PAGES.keys()])
    html = render_html(latest_day, ranked, sources_text)

    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    with open("docs/.nojekyll", "w", encoding="utf-8") as f:
        f.write("")

if __name__ == "__main__":
    main()
