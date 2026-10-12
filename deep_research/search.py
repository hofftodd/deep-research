"""Web search providers and page fetching/text extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from .config import Settings


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


class SearchError(RuntimeError):
    pass


# Sites that rarely yield extractable article text, or need a login.
SKIP_DOMAINS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "facebook.com", "www.facebook.com",
    "instagram.com", "www.instagram.com", "tiktok.com", "www.tiktok.com", "x.com", "twitter.com",
    "pinterest.com", "www.pinterest.com", "linkedin.com", "www.linkedin.com",
}
SKIP_EXTENSIONS = (".mp4", ".mp3", ".zip", ".exe", ".dmg", ".jpg", ".jpeg", ".png", ".gif")


def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication: drop fragments and common tracking params."""
    p = urlparse(url.strip())
    query = "&".join(
        part for part in p.query.split("&")
        if part and not part.lower().startswith(("utm_", "fbclid", "gclid", "ref="))
    )
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", query, ""))


def is_fetchable(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if p.netloc.lower() in SKIP_DOMAINS:
        return False
    return not p.path.lower().endswith(SKIP_EXTENSIONS)


class Searcher:
    def __init__(self, settings: Settings):
        self.s = settings
        self._http = httpx.Client(
            timeout=settings.fetch_timeout,
            headers={"User-Agent": settings.user_agent, "Accept-Language": "en-US,en;q=0.9"},
            follow_redirects=True,
        )

    def search(self, query: str, limit: int | None = None) -> list[SearchResult]:
        limit = limit or self.s.search_results_per_query
        provider = self.s.search_provider
        if provider == "searxng":
            results = self._searxng(query)
        elif provider == "brave":
            results = self._brave(query)
        elif provider == "duckduckgo":
            results = self._duckduckgo(query)
        else:
            raise SearchError(f"Unknown SEARCH_PROVIDER {provider!r}")
        seen: set[str] = set()
        out = []
        for r in results:
            key = normalize_url(r.url)
            if key in seen or not is_fetchable(r.url):
                continue
            seen.add(key)
            out.append(r)
        return out[:limit]

    def _searxng(self, query: str) -> list[SearchResult]:
        try:
            resp = self._http.get(f"{self.s.searxng_url}/search", params={"q": query, "format": "json"})
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"SearXNG search failed (is JSON output enabled?): {exc}") from exc
        return [
            SearchResult(r.get("url", ""), r.get("title", ""), r.get("content", ""))
            for r in data.get("results", []) if r.get("url")
        ]

    def _brave(self, query: str) -> list[SearchResult]:
        if not self.s.brave_api_key:
            raise SearchError("SEARCH_PROVIDER=brave requires BRAVE_API_KEY")
        try:
            resp = self._http.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 20},
                headers={"X-Subscription-Token": self.s.brave_api_key, "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"Brave search failed: {exc}") from exc
        return [
            SearchResult(r["url"], r.get("title", ""), BeautifulSoup(r.get("description", ""), "html.parser").get_text())
            for r in data.get("web", {}).get("results", []) if r.get("url")
        ]

    def _duckduckgo(self, query: str) -> list[SearchResult]:
        try:
            resp = self._http.post("https://html.duckduckgo.com/html/", data={"q": query})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError(f"DuckDuckGo search failed: {exc}") from exc
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for res in soup.select("div.result"):
            if "result--ad" in (res.get("class") or []):
                continue
            a = res.select_one("a.result__a")
            if not a or not a.get("href"):
                continue
            href = a["href"]
            # DDG wraps links in a redirect: //duckduckgo.com/l/?uddg=<encoded url>
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            elif href.startswith("//"):
                href = "https:" + href
            snippet = res.select_one(".result__snippet")
            results.append(SearchResult(href, a.get_text(" ", strip=True),
                                        snippet.get_text(" ", strip=True) if snippet else ""))
        if not results and "anomaly" in resp.text.lower():
            raise SearchError("DuckDuckGo is rate-limiting requests; consider SEARCH_PROVIDER=searxng")
        return results


@dataclass
class Page:
    url: str
    title: str
    text: str


class FetchError(RuntimeError):
    pass


_NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form",
               "iframe", "svg", "button", "figure", "template"]
_NOISE_HINTS = re.compile(r"(cookie|consent|banner|newsletter|subscribe|sidebar|comment|share|social|promo|advert|related)", re.I)


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, main text) from an HTML document."""
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = og["content"].strip()
    for tag in soup(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.decomposed or tag.name in ("body", "html", "main", "article"):
            continue
        ident = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
        if not ident.strip() or not _NOISE_HINTS.search(ident):
            continue
        # Guard against layout wrappers like "page-with-sidebar" that hold the real content.
        if tag.find(["article", "main"]) or len(tag.get_text()) > 2000:
            continue
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "pre", "blockquote"]):
        # Skip containers whose text will be emitted by a nested block element.
        if el.find(["p", "li", "h1", "h2", "h3", "h4", "pre", "blockquote"]):
            continue
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if len(t) < 3:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            t = "\n## " + t
        blocks.append(t)
    text = "\n".join(blocks)
    if len(text) < 200:  # fall back to everything when the structure is unusual
        text = re.sub(r"\s+\n", "\n", root.get_text("\n", strip=True))
    return title, text.strip()


class Fetcher:
    def __init__(self, settings: Settings):
        self.s = settings
        self._http = httpx.Client(
            timeout=settings.fetch_timeout,
            headers={"User-Agent": settings.user_agent, "Accept-Language": "en-US,en;q=0.9",
                     "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5"},
            follow_redirects=True,
        )

    def fetch(self, url: str) -> Page:
        try:
            resp = self._http.get(url)
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise FetchError(f"HTTP {resp.status_code}")
        ctype = resp.headers.get("content-type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            raise FetchError("PDF documents are not supported")
        if "html" in ctype or "xml" in ctype:
            title, text = extract_text(resp.text)
        elif ctype.startswith("text/"):
            title, text = "", resp.text
        else:
            raise FetchError(f"Unsupported content type {ctype or 'unknown'}")
        if len(text) < 200:
            raise FetchError("Page had too little readable text")
        return Page(url=str(resp.url), title=title, text=text[: self.s.max_page_chars])
