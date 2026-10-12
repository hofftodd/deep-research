from deep_research import report as rpt


def test_drop_invalid_citations_keeps_only_known_ids():
    text = "Fact one [S1]. Fact two [S2, S99]. Fact three [S42]."
    out = rpt.drop_invalid_citations(text, {1, 2})
    assert out == "Fact one [S1]. Fact two [S2]. Fact three."


def test_citation_variants_are_recognised():
    assert rpt.extract_source_ids("a [S1] b [s2; S3] c [4] d [5,6]") == {1, 2, 3, 4, 5, 6}


def test_enforce_citations_drops_uncited_paragraphs_and_list_items():
    text = "## Heading\n\nCited paragraph [S1].\n\nUncited paragraph.\n\n- item one [S2]\n- item two"
    out, n = rpt.enforce_citations(text, "drop")
    assert n == 2
    assert "Uncited" not in out and "item two" not in out
    assert "Cited paragraph [S1]." in out and "- item one [S2]" in out


def test_enforce_citations_flag_policy():
    out, n = rpt.enforce_citations("No source here.", "flag")
    assert n == 1 and "[citation needed]" in out


def test_enforce_citations_removes_empty_headings():
    out, _ = rpt.enforce_citations("## Empty\n\nNothing cited.\n\n## Kept\n\nCited [S1].", "drop")
    assert "Empty" not in out and "## Kept" in out


def test_renumber_by_first_appearance_and_build_refs():
    sources = {
        7: {"url": "https://a.example/x", "title": "A", "fetched_at": "2026-01-02T00:00:00"},
        3: {"url": "https://www.b.example/y", "title": "", "fetched_at": "2026-01-03T00:00:00"},
    }
    text, refs = rpt.renumber("First [S7]. Second [S3, S7]. Third [S3].", sources)
    assert text == "First [1]. Second [1, 2]. Third [2]."
    assert [r["source_id"] for r in refs] == [7, 3]
    assert refs[1]["title"] == "www.b.example" and refs[1]["site"] == "b.example"
    md = rpt.references_markdown(refs)
    assert "1. [A](https://a.example/x)" in md


def test_render_html_escapes_model_html_and_links_citations():
    refs = [{"n": 1, "url": "https://a.example/?q=<x>", "title": "A <b>", "site": "a.example", "accessed": "2026-01-01"}]
    out = rpt.render_html("Claim <script>alert(1)</script> [1].", refs)
    assert "<script>" not in out
    assert 'href="#ref-1"' in out and 'id="ref-1"' in out
    assert "A &lt;b&gt;" in out


def test_trailing_uncited_sentence_is_removed():
    out, n = rpt.enforce_citations("Abundant [S1]. Costs may undercut LFP cells soon.", "drop")
    assert n == 1 and out == "Abundant [S1]."


def test_sentences_before_a_citation_are_kept():
    text = "Sodium is common. It is cheap to mine [S1]."
    assert rpt.enforce_citations(text, "drop") == (text, 0)


def test_text_directly_under_heading_is_checked():
    out, n = rpt.enforce_citations("## Heading\nUncited claim right under the heading.\n\nCited [S1].", "drop")
    assert n == 1 and "Uncited" not in out and out.startswith("## Heading\n\nCited")


def test_tables_are_not_trimmed():
    table = "| a | b |\n|---|---|\n| x [S1] | y |"
    out, n = rpt.enforce_citations(table, "drop")
    assert out == table and n == 0


def test_render_html_neutralises_dangerous_links_and_images():
    body = (
        "A [js](javascript:alert(1)) link [1]. "
        "An [encoded](javascript&#58;alert(1)) one. "
        "A [spaced]( java\tscript:alert(1)) one. "
        "A [data](data:text/html,<b>x</b>) one. "
        "An image ![x](https://evil.example/x.png). "
        "A [good](https://ok.example/page) link."
    )
    out = rpt.render_html(body, [])
    assert "javascript" not in out.lower() and "data:" not in out and "<img" not in out
    assert 'href="https://ok.example/page" target="_blank" rel="noopener noreferrer nofollow"' in out
    assert "js" in out and "encoded" in out  # link text survives as plain text
    assert 'href="#ref-1"' in out


def test_reference_with_unsafe_url_is_not_linked():
    refs = [{"n": 1, "url": "javascript:alert(1)", "title": "T", "site": "x", "accessed": "2026-01-01"}]
    out = rpt.render_html("Claim [1].", refs)
    assert "javascript" not in out and 'id="ref-1"' in out


def test_safe_url():
    assert rpt.safe_url("https://a.example/x?y=1&amp;z=2")
    assert rpt.safe_url("#ref-3")
    for bad in ("javascript:x", "JAVASCRIPT:x", "javascript&amp;#58;x", "vbscript:x", "data:x", "/relative", "//evil.example"):
        assert rpt.safe_url(bad) is None, bad


# ---- citation units ----------------------------------------------------------------------------
def test_citation_units_group_preceding_sentences_and_adjacent_citations():
    text = "## H\n\nSodium is common. It is cheap [S1]. Grid favours cost [S2][S3].\n\n| a | b [S1] |\n|---|---|"
    units = rpt.citation_units(text)
    assert [(u.statement, u.source_ids) for u in units] == [
        ("Sodium is common. It is cheap.", [1]),
        ("Grid favours cost.", [2, 3]),
    ]  # the table is not split into units


def test_rewrite_units_drops_recites_and_cleans_up():
    text = "## H\n\nKeep this [S1]. Drop this [S2].\n\n- dropped item [S3]\n- kept item [S4]\n\n## Empty\n\nGone [S5]."

    def fn(u):
        if "Drop" in u.text or "dropped" in u.text or "Gone" in u.text:
            return ""
        if "kept item" in u.text:
            return rpt.recite(u.text, [7, 8])
        return u.text

    assert rpt.rewrite_units(text, fn) == "## H\n\nKeep this [S1].\n\n- kept item [S7, S8]"


def test_recite_keeps_final_punctuation():
    assert rpt.recite("It rose 5% in 2024 [S1][S2].", [3]) == "It rose 5% in 2024 [S3]."
    assert rpt.recite("No punctuation [S1]", [2]) == "No punctuation [S2]"
