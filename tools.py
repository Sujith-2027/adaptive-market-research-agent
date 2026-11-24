"""Web tools: search (Tavily if a key exists, else keyless DuckDuckGo) and page reader. Both cached."""
import os, json, hashlib, pathlib
import requests
from bs4 import BeautifulSoup

CACHE = pathlib.Path(".cache")
CACHE.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (research-agent; educational project)"}


def _file(kind, s):
    return CACHE / f"{kind}_{hashlib.sha256(s.encode()).hexdigest()}.json"


def search(query, n=3):
    f = _file("search", query)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    results = []
    if os.getenv("TAVILY_API_KEY"):
        try:
            from tavily import TavilyClient
            r = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(query, max_results=n)["results"]
            results = [{"title": x["title"], "url": x["url"], "content": x["content"]} for x in r]
        except Exception:
            results = []
    if not results:
        try:
            from ddgs import DDGS
            results = [{"title": h["title"], "url": h["href"], "content": h["body"]}
                       for h in DDGS().text(query, max_results=n)]
        except Exception:
            results = []
    if results:
        f.write_text(json.dumps(results), encoding="utf-8")
    return results


def read(url, limit=2000):
    f = _file("page", url)
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))[:limit]
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        if "html" not in r.headers.get("content-type", ""):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())
        f.write_text(json.dumps(text[:8000]), encoding="utf-8")
        return text[:limit]
    except Exception:
        return ""
