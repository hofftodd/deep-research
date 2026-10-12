"""Citation handling for generated reports.

The model is asked to cite source ids like [S12]. Everything here is deterministic code so the
final report can only ever reference sources that were actually fetched and used:

* citations to unknown source ids are removed,
* citations are renumbered [1], [2], ... in order of first appearance,
* paragraphs without any citation are dropped (or flagged),
* text can be split into citation units (statement + its citation) so each one can be checked
  against its sources and kept, re-cited or removed, and
* the References section is generated from the database, never by the model.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse

import markdown as md

# Matches [S3], [S3, S7], [S3; S7], [3], [3,4], [S3][S4] is matched as two groups.
CITE_RE = re.compile(r"\[\s*S?\d+(?:\s*[,;]\s*S?\d+)*\s*\]", re.IGNORECASE)
NUM_RE = re.compile(r"\d+")
FINAL_CITE_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")


def extract_source_ids(text: str) -> set[int]:
    ids: set[int] = set()
    for m in CITE_RE.finditer(text):
        ids.update(int(n) for n in NUM_RE.findall(m.group(0)))
    return ids


def drop_invalid_citations(text: str, valid_ids: set[int]) -> str:
    """Remove citation ids the model invented; keep the valid ones as [S#, S#]."""

    def repl(m: re.Match) -> str:
        ids = [int(n) for n in NUM_RE.findall(m.group(0))]
        kept = [i for i in dict.fromkeys(ids) if i in valid_ids]
        return "[" + ", ".join(f"S{i}" for i in kept) + "]" if kept else ""

    text = CITE_RE.sub(repl, text)
    return re.sub(r"[ \t]+([.,;:])", r"\1", text)


def _is_structural(block: str) -> bool:
    """Blocks that carry no factual claims and therefore need no citation."""
    s = block.strip()
    return not s or s.startswith("#") or s.startswith("|---") or set(s) <= set("-*_ ")


def _split_items(block: str) -> list[str]:
    """Split a markdown block into paragraphs/list items so each is checked separately."""
    lines = block.split("\n")
    if all(re.match(r"\s*([-*+]|\d+[.)])\s+", ln) or not ln.strip() or ln.startswith("  ") for ln in lines):
        items, cur = [], []
        for ln in lines:
            if re.match(r"\s*([-*+]|\d+[.)])\s+", ln) and cur:
                items.append("\n".join(cur))
                cur = []
            cur.append(ln)
        if cur:
            items.append("\n".join(cur))
        return items
    return [block]


def _trim_uncited_tail(item: str) -> tuple[str, bool]:
    """Remove trailing sentence(s) that come after an item's last citation."""
    if item.lstrip().startswith("|"):  # tables: leave rows intact
        return item, False
    last = None
    for last in CITE_RE.finditer(item):
        pass
    if last is None:
        return item, False
    tail = item[last.end():]
    lead = re.match(r"[\s.,;:!?)\]*_]*", tail).group(0)
    if len(re.findall(r"[A-Za-z]", tail[len(lead):])) < 12:
        return item, False
    return item[: last.end()] + lead.rstrip(), True


def enforce_citations(text: str, policy: str = "drop") -> tuple[str, int]:
    """Drop (or flag) paragraphs, list items and trailing sentences that carry no citation.

    A sentence counts as cited if it, or a later sentence in the same paragraph, ends with a
    citation (models often cite once at the end of a run of sentences from one source).
    Returns the new text and the number of uncited passages found.
    """
    # Make sure headings are their own blocks so text directly beneath them is still checked.
    text = re.sub(r"^(#{1,6}[ \t].*)$", r"\n\1\n", text.strip(), flags=re.MULTILINE)
    out_blocks = []
    uncited = 0
    for block in re.split(r"\n\s*\n", text.strip()):
        if _is_structural(block):
            out_blocks.append(block.strip())
            continue
        kept = []
        for item in _split_items(block):
            if CITE_RE.search(item):
                trimmed, cut = _trim_uncited_tail(item)
                if cut:
                    uncited += 1
                    if policy == "flag":
                        trimmed = item.rstrip() + " *[citation needed]*"
                kept.append(trimmed)
                continue
            if len(item.strip()) < 3 or _is_structural(item):
                kept.append(item)
                continue
            uncited += 1
            if policy == "flag":
                kept.append(item.rstrip() + " *[citation needed]*")
        if kept:
            out_blocks.append("\n".join(kept))
    return _join_blocks(out_blocks), uncited


def _heading_level(block: str) -> int:
    return len(block.strip()) - len(block.strip().lstrip("#"))


def _join_blocks(blocks: list[str]) -> str:
    """Join blocks, removing headings left with nothing beneath them."""
    cleaned = []
    for i, block in enumerate(blocks):
        if block.strip().startswith("#"):
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if nxt is None or nxt.strip().startswith("#") and _heading_level(nxt) <= _heading_level(block):
                continue
        cleaned.append(block)
    return "\n\n".join(cleaned)


# ---- citation units -------------------------------------------------------------------------
# A unit is a run of text ending in a citation group, plus the punctuation right after it:
# "A is true. B follows from it [S3]." is one unit citing S3. Uncited sentences before a citation
# therefore belong to that citation's unit and get checked with it.

_TRAIL_RE = re.compile(r"[.,;:!?)\]\"'\u201d\u2019]*")
_LIST_PREFIX_RE = re.compile(r"\s*(?:[-*+]|\d+[.)])\s+")


@dataclass
class Unit:
    id: int
    text: str           # the original text of the unit, citation included
    source_ids: list[int]
    statement: str      # the text without citation tags


def _segment_item(item: str) -> tuple[str, list[str], str]:
    """Split one paragraph/list item into (list prefix, units, uncited tail)."""
    m = _LIST_PREFIX_RE.match(item)
    prefix = m.group(0) if m else ""
    body = item[len(prefix):]
    segs: list[str] = []
    pos = 0
    for c in CITE_RE.finditer(body):
        end = c.end() + len(_TRAIL_RE.match(body, c.end()).group(0))
        if segs and not body[pos:c.start()].strip():
            segs[-1] += body[pos:end]  # adjacent groups like [S1][S2] belong to one statement
        else:
            segs.append(body[pos:end])
        pos = end
    return prefix, segs, body[pos:]


def _parse_units(text: str) -> list[str | list[tuple[str, list[str], str]]]:
    blocks: list[str | list[tuple[str, list[str], str]]] = []
    text = re.sub(r"^(#{1,6}[ \t].*)$", r"\n\1\n", text.strip(), flags=re.MULTILINE)
    for block in re.split(r"\n\s*\n", text.strip()):
        if _is_structural(block) or block.lstrip().startswith("|"):
            blocks.append(block.strip())  # headings, rules and tables are kept as they are
        else:
            blocks.append([_segment_item(item) for item in _split_items(block)])
    return blocks


def strip_citations(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s{2,}", " ", CITE_RE.sub("", text))).strip()


def citation_units(text: str) -> list[Unit]:
    units: list[Unit] = []
    for block in _parse_units(text):
        if isinstance(block, str):
            continue
        for _, segs, _ in block:
            for seg in segs:
                units.append(Unit(len(units) + 1, seg, sorted(extract_source_ids(seg)), strip_citations(seg)))
    return units


def rewrite_units(text: str, fn: Callable[[Unit], str]) -> str:
    """Rebuild *text* with each citation unit replaced by fn(unit); "" removes the unit.

    Units are numbered exactly as citation_units() numbers them. Items, paragraphs and headings
    left empty are removed."""
    units = iter(citation_units(text))
    out_blocks = []
    for block in _parse_units(text):
        if isinstance(block, str):
            out_blocks.append(block)
            continue
        items = []
        for prefix, segs, tail in block:
            if not segs:
                items.append(prefix + tail)
                continue
            kept = [r.strip() for r in (fn(next(units)) for _ in segs) if r and r.strip()]
            if kept:
                items.append(prefix + " ".join(kept) + tail.rstrip())
        if items:
            out_blocks.append("\n".join(items))
    return _join_blocks(out_blocks)


def recite(unit_text: str, source_ids: list[int]) -> str:
    """Replace a unit's citation(s) with one citation of *source_ids*, before final punctuation."""
    body = strip_citations(unit_text)
    m = re.search(r"[.,;:!?]*$", body)
    cite = "[" + ", ".join(f"S{i}" for i in source_ids) + "]"
    return body[: m.start()].rstrip() + " " + cite + body[m.start():]


def renumber(text: str, sources: dict[int, dict]) -> tuple[str, list[dict]]:
    """Rewrite [S12, S4] -> [1, 2] numbered by first appearance and build the reference list."""
    order: dict[int, int] = {}

    def repl(m: re.Match) -> str:
        nums = []
        for n in NUM_RE.findall(m.group(0)):
            sid = int(n)
            if sid not in sources:
                continue
            if sid not in order:
                order[sid] = len(order) + 1
            nums.append(order[sid])
        nums = sorted(dict.fromkeys(nums))
        return "[" + ", ".join(map(str, nums)) + "]" if nums else ""

    text = CITE_RE.sub(repl, text)
    refs = []
    for sid, num in sorted(order.items(), key=lambda kv: kv[1]):
        s = sources[sid]
        refs.append({
            "n": num,
            "source_id": sid,
            "url": s["url"],
            "title": (s.get("title") or "").strip() or urlparse(s["url"]).netloc,
            "site": urlparse(s["url"]).netloc.removeprefix("www."),
            "accessed": (s.get("fetched_at") or datetime.now(timezone.utc).isoformat())[:10],
        })
    return text, refs


def references_markdown(refs: list[dict]) -> str:
    lines = ["## References", ""]
    for r in refs:
        title = r["title"].replace("[", "(").replace("]", ")")
        lines.append(f"{r['n']}. [{title}]({r['url']}) — {r['site']}, accessed {r['accessed']}")
    return "\n".join(lines)


_LINK_RE = re.compile(r'<a href="([^"]*)"([^>]*)>(.*?)</a>', re.DOTALL)
_IMG_RE = re.compile(r"<img\b[^>]*>")
_CTRL_RE = re.compile(r"[\x00-\x20\x7f]+")


def safe_url(url: str) -> str | None:
    """Return the URL if it is an http(s) link or an in-page anchor, else None.

    Entities are decoded first (repeatedly) because browsers decode them in attributes, so
    `javascript&#58;...` must be caught too.
    """
    prev, cur = None, url
    while prev != cur:
        prev, cur = cur, html.unescape(cur)
    cur = _CTRL_RE.sub("", cur).lower()
    if cur.startswith(("http://", "https://")) or re.fullmatch(r"#[\w-]+", cur):
        return url
    return None


def _sanitize_links(body: str) -> str:
    """Model-written text can carry Markdown links (possibly planted by a web page via prompt
    injection). Keep only http(s) links, opened in a new tab; reduce anything else to its text."""

    def repl(m: re.Match) -> str:
        href, text = m.group(1), m.group(3)
        if safe_url(href) is None:
            return text
        if href.startswith("#"):
            return f'<a href="{href}">{text}</a>'
        return f'<a href="{href}" target="_blank" rel="noopener noreferrer nofollow">{text}</a>'

    return _IMG_RE.sub("", _LINK_RE.sub(repl, body))


def render_html(body_md: str, refs: list[dict]) -> str:
    """Render the report body (model-written, so raw HTML is escaped and links are sanitized)
    plus linked references."""
    safe = body_md.replace("&", "&amp;").replace("<", "&lt;")
    body = _sanitize_links(md.markdown(safe, extensions=["tables", "sane_lists"]))

    def link(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        return "<sup class=\"cite\">[" + ", ".join(
            f'<a href="#ref-{p}" title="Source {p}">{p}</a>' for p in parts
        ) + "]</sup>"

    body = FINAL_CITE_RE.sub(link, body)
    if not refs:
        return body
    items = "".join(
        f'<li id="ref-{r["n"]}" value="{r["n"]}">'
        + (f'<a href="{html.escape(r["url"])}" target="_blank" rel="noopener noreferrer">{html.escape(r["title"])}</a>'
           if safe_url(r["url"]) else html.escape(r["title"]))
        + f' <span class="ref-meta">— {html.escape(r["site"])}, accessed {r["accessed"]}</span></li>'
        for r in refs
    )
    return body + f'<h2>References</h2><ol class="references">{items}</ol>'
