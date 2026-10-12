"""Search providers and the page fetcher against faked HTTP responses."""

import httpx
import pytest

from deep_research.config import Settings
from deep_research.search import Fetcher, FetchError, Searcher, SearchError

ARTICLE = "<html><head><title>Article</title></head><body><article><p>%s</p></article></body></html>" % (
    "Useful readable sentence about the topic. " * 20)


def searcher(handler, **settings):
    return Searcher(Settings(**settings), transport=httpx.MockTransport(handler))


def fetcher(handler, **settings):
    return Fetcher(Settings(**settings), transport=httpx.MockTransport(handler))


# ---- SearXNG ---------------------------------------------------------------------------------
def test_searxng_parses_dedupes_and_filters():
    seen = {}

    def handler(request):
        seen["url"] = request.url
        return httpx.Response(200, json={"results": [
            {"url": "https://a.example/x?utm_source=feed", "title": "A", "content": "snippet a"},
            {"url": "https://a.example/x", "title": "A again", "content": ""},
            {"url": "https://www.youtube.com/watch?v=1", "title": "video"},
            {"url": "https://b.example/file.zip", "title": "zip"},
            {"title": "no url"},
            {"url": "https://c.example/y", "title": "C", "content": "snippet c"},
        ]})

    results = searcher(handler, search_provider="searxng", searxng_url="http://sx.test").search("q one")
    assert [r.title for r in results] == ["A", "C"]
    assert results[0].snippet == "snippet a"
    assert seen["url"].path == "/search" and seen["url"].params["format"] == "json"
    assert seen["url"].params["q"] == "q one"


def test_search_respects_limit():
    handler = lambda r: httpx.Response(200, json={"results": [
        {"url": f"https://s{i}.example/", "title": str(i)} for i in range(20)]})
    assert len(searcher(handler, search_provider="searxng").search("q", limit=5)) == 5


@pytest.mark.parametrize("response", [
    httpx.Response(403, text="forbidden"),
    httpx.Response(200, text="<html>not json</html>"),
])
def test_searxng_errors(response):
    with pytest.raises(SearchError, match="SearXNG"):
        searcher(lambda r: response, search_provider="searxng").search("q")


# ---- Brave ---------------------------------------------------------------------------------------
def test_brave_parses_and_sends_key():
    seen = {}

    def handler(request):
        seen["key"] = request.headers.get("X-Subscription-Token")
        return httpx.Response(200, json={"web": {"results": [
            {"url": "https://a.example/", "title": "A", "description": "has <strong>bold</strong> text"},
        ]}})

    results = searcher(handler, search_provider="brave", brave_api_key="k123").search("q")
    assert seen["key"] == "k123"
    assert results[0].snippet == "has bold text"


def test_brave_requires_key_and_reports_errors():
    with pytest.raises(SearchError, match="BRAVE_API_KEY"):
        searcher(lambda r: httpx.Response(200), search_provider="brave").search("q")
    with pytest.raises(SearchError, match="Brave"):
        searcher(lambda r: httpx.Response(429), search_provider="brave", brave_api_key="k").search("q")


# ---- DuckDuckGo ----------------------------------------------------------------------------------
DDG_HTML = """
<div class="result result--ad"><a class="result__a" href="https://ads.example/">Ad</a></div>
<div class="result"><a class="result__a"
   href="//duckduckgo.com/l/?uddg=https%3A%2F%2Freal.example%2Fpage%3Fid%3D1&amp;rut=abc">Real page</a>
   <a class="result__snippet">The snippet</a></div>
<div class="result"><a class="result__a" href="//plain.example/x">Plain</a></div>
<div class="result"><span>no link</span></div>
"""


def test_duckduckgo_parses_redirects_and_skips_ads():
    seen = {}

    def handler(request):
        seen["method"], seen["body"] = request.method, request.content
        return httpx.Response(200, text=DDG_HTML)

    results = searcher(handler, search_provider="duckduckgo").search("sodium ion")
    assert [(r.url, r.title) for r in results] == [
        ("https://real.example/page?id=1", "Real page"),
        ("https://plain.example/x", "Plain"),
    ]
    assert results[0].snippet == "The snippet"
    assert seen["method"] == "POST" and b"q=sodium+ion" in seen["body"]


def test_duckduckgo_rate_limit_is_reported():
    page = "<html><body>If this error persists... anomaly detected</body></html>"
    with pytest.raises(SearchError, match="rate-limiting"):
        searcher(lambda r: httpx.Response(202, text=page), search_provider="duckduckgo").search("q")


def test_duckduckgo_http_error():
    with pytest.raises(SearchError, match="DuckDuckGo"):
        searcher(lambda r: httpx.Response(503), search_provider="duckduckgo").search("q")


def test_unknown_provider():
    with pytest.raises(SearchError, match="Unknown"):
        searcher(lambda r: httpx.Response(200), search_provider="bing").search("q")


# ---- Fetcher -------------------------------------------------------------------------------------
def test_fetch_html_follows_redirects_and_truncates():
    def handler(request):
        if request.url.path == "/old":
            return httpx.Response(301, headers={"Location": "https://site.example/new"})
        return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html; charset=utf-8"})

    page = fetcher(handler, max_page_chars=300).fetch("https://site.example/old")
    assert page.url == "https://site.example/new"
    assert page.title == "Article" and len(page.text) == 300
    assert page.text.startswith("Useful readable sentence")


def test_fetch_plain_text():
    text = "Plain text document. " * 20
    page = fetcher(lambda r: httpx.Response(200, text=text, headers={"content-type": "text/plain"})).fetch("https://t.example/")
    assert page.text == text and page.title == ""


@pytest.mark.parametrize("response,url,message", [
    (httpx.Response(404), "https://x.example/", "HTTP 404"),
    (httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}), "https://x.example/r", "PDF"),
    (httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"}), "https://x.example/r.pdf", "PDF"),
    (httpx.Response(200, content=b"\x00", headers={"content-type": "image/png"}), "https://x.example/", "Unsupported"),
    (httpx.Response(200, text="<p>tiny</p>", headers={"content-type": "text/html"}), "https://x.example/", "too little"),
])
def test_fetch_rejections(response, url, message):
    with pytest.raises(FetchError, match=message):
        fetcher(lambda r: response).fetch(url)


def test_fetch_network_error():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(FetchError, match="ConnectError"):
        fetcher(handler).fetch("https://down.example/")
