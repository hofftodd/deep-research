from deep_research.agent import chunk_text, quote_supported
from deep_research.search import extract_text, is_fetchable, normalize_url


def test_normalize_url_strips_tracking_and_fragments():
    assert normalize_url("HTTPS://Example.com/a/?utm_source=x&id=3#frag") == "https://example.com/a?id=3"


def test_is_fetchable():
    assert is_fetchable("https://example.com/article")
    assert not is_fetchable("https://www.youtube.com/watch?v=1")
    assert not is_fetchable("ftp://example.com/file")


def test_extract_text_prefers_article_and_drops_noise():
    html = """<html><head><title>T</title></head><body>
      <nav>Home | About</nav>
      <div class="layout-with-sidebar"><article><h1>Heading</h1>
        <p>%s</p><div class="share-buttons">Share on X</div></article></div>
      <footer>Copyright</footer><script>var x=1;</script></body></html>""" % ("Real content. " * 30)
    title, text = extract_text(html)
    assert title == "T"
    assert "Real content." in text and "## Heading" in text
    assert "Home | About" not in text and "Share on X" not in text and "var x" not in text


def test_quote_supported():
    page = "The plant produced 4.2 TWh of electricity in 2023, a record for the site."
    assert quote_supported("produced 4.2 TWh of electricity in 2023", page)
    assert quote_supported("The plant produced 4.2 TWh of electricity in 2023 - a record", page)
    assert not quote_supported("The plant was closed permanently in 2019 after protests", page)
    assert not quote_supported("short", page)


def test_chunk_text():
    text = ("line\n" * 1000)
    chunks = chunk_text(text, 1000, 3)
    assert len(chunks) == 3 and all(len(c) <= 1000 for c in chunks)
