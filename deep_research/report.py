"""Citation handling for generated reports.

The model is asked to cite source ids like [S12]. Everything here is deterministic code so the
final report can only ever reference sources that were actually fetched and used:

* citations to unknown source ids are removed,
* citations are renumbered [1], [2], ... in order of first appearance,
* paragraphs without any citation are dropped (or flagged), and
* the References section is generated from the database, never by the model.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
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
    # Remove headings left with nothing beneath them.
    cleaned = []
    for i, block in enumerate(out_blocks):
        if block.strip().startswith("#"):
            nxt = out_blocks[i + 1] if i + 1 < len(out_blocks) else None
            if nxt is None or nxt.strip().startswith("#") and _heading_level(nxt) <= _heading_level(block):
                continue
        cleaned.append(block)
    return "\n\n".join(cleaned), uncited


def _heading_level(block: str) -> int:
    return len(block.strip()) - len(block.strip().lstrip("#"))


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


def render_html(body_md: str, refs: list[dict]) -> str:
    """Render the report body (model-written, so raw HTML is escaped) plus linked references."""
    safe = body_md.replace("&", "&amp;").replace("<", "&lt;")
    body = md.markdown(safe, extensions=["tables", "sane_lists"])

    def link(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        return "<sup class=\"cite\">[" + ", ".join(
            f'<a href="#ref-{p}" title="Source {p}">{p}</a>' for p in parts
        ) + "]</sup>"

    body = FINAL_CITE_RE.sub(link, body)
    if not refs:
        return body
    items = "".join(
        f'<li id="ref-{r["n"]}" value="{r["n"]}"><a href="{html.escape(r["url"])}" target="_blank" rel="noopener noreferrer">'
        f'{html.escape(r["title"])}</a> <span class="ref-meta">— {html.escape(r["site"])}, accessed {r["accessed"]}</span></li>'
        for r in refs
    )
    return body + f'<h2>References</h2><ol class="references">{items}</ol>'
