"""The research agent: plan -> (search -> read -> take notes -> reflect)* -> write cited report."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable
from urllib.parse import urlparse

from . import report as rpt
from .config import Settings
from .db import Database, now
from .llm import LLMClient, LLMError
from .search import Fetcher, FetchError, Searcher, SearchError, SearchResult, normalize_url

SYSTEM = (
    "You are a meticulous research assistant. You value accuracy over fluency, never invent "
    "facts, and keep track of exactly which source supports each statement. Today's date is {today}."
)


class Cancelled(Exception):
    pass


class ModelUnavailable(RuntimeError):
    """Too many model calls failed in a row; the run cannot usefully continue.

    Deliberately not an LLMError, so the per-step fallbacks don't swallow it."""


@dataclass
class Options:
    max_iterations: int
    max_questions: int
    max_sources: int
    sources_per_question: int
    queries_per_question: int

    @classmethod
    def resolve(cls, settings: Settings, overrides: dict[str, Any]) -> "Options":
        def pick(name: str, lo: int, hi: int) -> int:
            val = overrides.get(name)
            val = getattr(settings, name) if val in (None, "") else int(val)
            return max(lo, min(hi, val))

        return cls(
            max_iterations=pick("max_iterations", 1, 10),
            max_questions=pick("max_questions", 1, 12),
            max_sources=pick("max_sources", 1, 100),
            sources_per_question=pick("sources_per_question", 1, 10),
            queries_per_question=pick("queries_per_question", 1, 5),
        )


def _clean_list(value: Any, limit: int) -> list[str]:
    if isinstance(value, dict):  # e.g. {"questions": [...]} nested one level deeper
        value = next((v for v in value.values() if isinstance(v, list)), [])
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if isinstance(v, dict):
            v = v.get("question") or v.get("query") or v.get("text") or ""
        v = str(v).strip().strip('"')
        if v and v.lower() not in (o.lower() for o in out):
            out.append(v)
    return out[:limit]


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def quote_supported(quote: str, page_text: str, norm_page: str | None = None) -> bool:
    """Check the model's supporting quote really appears (near-verbatim) in the page text."""
    q = _norm_text(quote)
    if len(q) < 12:
        return False
    norm_page = norm_page if norm_page is not None else _norm_text(page_text)
    if q in norm_page:
        return True
    # Tolerate light paraphrase: most 4-word shingles of the quote must appear in the page.
    words = q.split()
    if len(words) < 6:
        return False
    shingles = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    hits = sum(1 for sh in shingles if sh in norm_page)
    return hits / len(shingles) >= 0.6


def chunk_text(text: str, size: int, max_chunks: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n", start + size // 2, end)
            end = cut if cut != -1 else end
        chunks.append(text[start:end])
        start = end
    return chunks


class ResearchAgent:
    def __init__(self, topic_id: int, db: Database, llm: LLMClient, searcher: Searcher,
                 fetcher: Fetcher, settings: Settings, cancel_event: threading.Event | None = None):
        self.topic_id = topic_id
        self.db = db
        self.llm = llm
        self.searcher = searcher
        self.fetcher = fetcher
        self.s = settings
        self.cancel_event = cancel_event or threading.Event()
        topic = db.get_topic(topic_id)
        assert topic is not None
        self.topic = topic
        self.opts = Options.resolve(settings, topic["options"])
        self.system = SYSTEM.format(today=date.today().isoformat())
        self._read_pairs: set[tuple[int, int]] = set()  # (source_id, question_id) already mined
        self._failures = 0  # consecutive failed model calls

    # ---- helpers ------------------------------------------------------------------------
    def log(self, message: str, level: str = "info") -> None:
        self.db.log(self.topic_id, message, level)

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def set_state(self, status: str | None = None, phase: str | None = None, progress: float | None = None) -> None:
        fields: dict[str, Any] = {}
        if status:
            fields["status"] = status
        if phase is not None:
            fields["phase"] = phase
        if progress is not None:
            fields["progress"] = round(max(0.0, min(100.0, progress)), 1)
        self.db.update_topic(self.topic_id, **fields)

    @property
    def subject(self) -> str:
        t = self.topic
        return t["title"] + (f"\n\nAdditional guidance from the requester:\n{t['description']}" if t["description"].strip() else "")

    def _call(self, fn: Callable[[], Any]) -> Any:
        """Run one model call. A failure raises LLMError for the caller to fall back on, unless
        too many calls have now failed in a row, which raises ModelUnavailable to end the run."""
        self.check_cancel()
        try:
            result = fn()
        except LLMError as exc:
            self._failures += 1
            self._flush_llm_warnings()
            if self._failures >= self.s.llm_max_consecutive_failures:
                raise ModelUnavailable(f"Stopped after {self._failures} model calls failed in a row. "
                                       f"Last error: {exc}") from exc
            raise
        self._failures = 0
        self._flush_llm_warnings()
        return result

    def _flush_llm_warnings(self) -> None:
        for w in self.llm.pop_warnings():
            self.log(w, "warning")

    def ask_json(self, prompt: str, mode: str = "fast") -> Any:
        return self._call(lambda: self.llm.chat_json(self.system, prompt, mode=mode))

    def ask_text(self, prompt: str, mode: str = "deep") -> str:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]
        return self._call(lambda: self.llm.chat(messages, mode=mode))

    def _log_usage(self) -> None:
        st = self.llm.stats
        if st["calls"]:
            self.log(f"Model usage: {st['calls']} call(s), {st['prompt_tokens']:,} prompt + "
                     f"{st['completion_tokens']:,} output tokens, {st['seconds'] / 60:.1f} min of model time.")

    # ---- pipeline -----------------------------------------------------------------------
    def run(self) -> None:
        self.db.update_topic(self.topic_id, started_at=now(), error=None)
        self.log(f"Starting research with model '{self.llm.model}' "
                 f"(up to {self.opts.max_iterations} rounds, {self.opts.max_sources} sources).")
        try:
            self.plan()
            for iteration in range(1, self.opts.max_iterations + 1):
                done = self.research_round(iteration)
                if done:
                    break
            self.write_report()
            self._log_usage()
        except Cancelled:
            self.set_state(status="cancelled", phase="Cancelled by user")
            self.log("Research cancelled.", "warning")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.set_state(status="failed", phase="Failed")
            self.db.update_topic(self.topic_id, error=str(exc))
            self.log(f"Research failed: {exc}", "error")
            self._log_usage()
            raise

    def plan(self) -> None:
        self.set_state(status="planning", phase="Breaking the topic into research questions", progress=2)
        try:
            data = self.ask_json(
                f"Research topic:\n{self.subject}\n\n"
                f"Break this topic into {self.opts.max_questions} or fewer focused, non-overlapping research "
                "questions that together would let someone write a thorough, balanced report. Cover background, "
                "key facts and figures, competing viewpoints, and recent developments where relevant.\n\n"
                'Return JSON: {"questions": ["...", "..."]}',
                mode="deep",
            )
        except LLMError as exc:
            data, reason = {}, f"Planning failed ({exc})"
        else:
            reason = "Model returned no plan"
        questions = _clean_list(data.get("questions") if isinstance(data, dict) else data, self.opts.max_questions)
        if not questions:
            questions = [self.topic["title"]]
            self.log(f"{reason}; researching the topic as a single question.", "warning")
        for q in questions:
            self.db.add_question(self.topic_id, q, iteration=1)
        self.log("Research plan:\n" + "\n".join(f"• {q}" for q in questions))

    def _progress(self, iteration: int, frac_in_round: float) -> float:
        # Planning 0-5%, research 5-80%, writing 80-100%.
        per_round = 75.0 / self.opts.max_iterations
        return 5 + per_round * (iteration - 1 + frac_in_round)

    def research_round(self, iteration: int) -> bool:
        """Research all open questions, then reflect. Returns True when research is sufficient."""
        open_qs = [q for q in self.db.questions(self.topic_id) if q["status"] == "open"]
        if not open_qs:
            return True
        self.set_state(status="researching", phase=f"Round {iteration}: researching {len(open_qs)} question(s)",
                       progress=self._progress(iteration, 0))
        for i, q in enumerate(open_qs):
            if self.db.count_sources(self.topic_id, "used") >= self.opts.max_sources:
                self.log("Source budget reached.")
                break
            self.set_state(phase=f"Round {iteration}: {q['text']}", progress=self._progress(iteration, i / len(open_qs)))
            self.research_question(q, iteration)
        self.set_state(phase=f"Round {iteration}: reviewing coverage", progress=self._progress(iteration, 0.95))
        return self.reflect(iteration)

    def research_question(self, q: dict, iteration: int) -> None:
        self.log(f"Researching: {q['text']}")
        prior = [n["claim"] for n in self.db.notes(self.topic_id, q["id"])][:10]
        try:
            data = self.ask_json(
                f"Overall topic: {self.topic['title']}\nResearch question: {q['text']}\n"
                + (f"Queries already tried: {q['queries']}\n" if q["queries"] else "")
                + (f"Already known: {prior}\n" if prior else "")
                + f"\nWrite {self.opts.queries_per_question} diverse web search queries (short, keyword-style, "
                'like a human would type into a search engine) to find authoritative sources.\n'
                'Return JSON: {"queries": ["...", "..."]}'
            )
        except LLMError as exc:
            self.log(f"Could not generate search queries ({exc}); searching for the question itself.", "warning")
            data = {}
        queries = _clean_list(data.get("queries") if isinstance(data, dict) else data, self.opts.queries_per_question)
        if not queries:
            queries = [q["text"]]
        self.db.update_question(q["id"], queries=q["queries"] + queries)

        # Map already-examined URLs to their source rows: pages read for an earlier question can be
        # mined again for this one without re-fetching or spending the source budget.
        known = {normalize_url(s["url"]): s for s in self.db.sources(self.topic_id)}
        candidates: list[SearchResult] = []
        seen: set[str] = set()
        for query in queries:
            self.check_cancel()
            try:
                results = self.searcher.search(query)
            except SearchError as exc:
                self.log(f"Search failed for '{query}': {exc}", "warning")
                continue
            self.log(f"Searched '{query}' → {len(results)} result(s)")
            for r in results:
                key = normalize_url(r.url)
                src = known.get(key)
                if key in seen or (src and (src["status"] == "failed" or (src["id"], q["id"]) in self._read_pairs)):
                    continue
                seen.add(key)
                candidates.append(r)

        candidates = self.rank_candidates(q, candidates)
        found = 0
        for r in candidates:
            if found >= self.opts.sources_per_question:
                break
            src = known.get(normalize_url(r.url))
            if not src and self.db.count_sources(self.topic_id, "used") >= self.opts.max_sources:
                break
            self.check_cancel()
            if self.read_source(q, r, src):
                found += 1
        self.db.update_question(q["id"], status="answered" if found else "unanswered")
        if not found:
            self.log(f"No usable sources found for: {q['text']}", "warning")

    def rank_candidates(self, q: dict, candidates: list[SearchResult]) -> list[SearchResult]:
        """Let the model pick the most promising results from titles/snippets; prefer domain diversity."""
        if len(candidates) <= self.opts.sources_per_question:
            return candidates
        listing = "\n".join(f"{i}. {c.title} — {c.url}\n   {c.snippet[:200]}" for i, c in enumerate(candidates[:20]))
        try:
            data = self.ask_json(
                f"Research question: {q['text']}\n\nSearch results:\n{listing}\n\n"
                "Rank the results most likely to contain substantive, reliable information answering the "
                "question (prefer primary sources, official bodies, reputable publications and research; avoid "
                "spam, forums and thin pages). Return JSON: {\"ranking\": [indices, best first]}"
            )
            raw = data.get("ranking") if isinstance(data, dict) else data
            order = [int(i) for i in raw if str(i).strip().lstrip("-").isdigit()]
        except (LLMError, TypeError, ValueError):
            order = []
        ranked = [candidates[i] for i in dict.fromkeys(order) if 0 <= i < min(20, len(candidates))]
        ranked += [c for c in candidates if c not in ranked]
        # Light domain diversity: push repeat domains to the back.
        seen_domains: set[str] = set()
        first, rest = [], []
        for c in ranked:
            dom = urlparse(c.url).netloc.lower().removeprefix("www.")
            (rest if dom in seen_domains else first).append(c)
            seen_domains.add(dom)
        return first + rest

    def read_source(self, q: dict, r: SearchResult, known: dict | None = None) -> bool:
        """Fetch (or reuse) a page and extract quote-verified notes for question *q*."""
        if known:
            source_id, title, url = known["id"], known["title"] or r.title, known["url"]
            text = self.db.source_content(source_id)
            if not text:
                return False
        else:
            source_id = self.db.add_source(self.topic_id, q["id"], r.url, r.title, r.snippet)
            if source_id is None:
                return False
            try:
                page = self.fetcher.fetch(r.url)
            except FetchError as exc:
                self.db.update_source(source_id, status="failed", error=str(exc), fetched_at=now())
                self.log(f"Could not read {r.url}: {exc}", "warning")
                return False
            title, url, text = page.title or r.title, r.url, page.text
            self.db.update_source(source_id, title=title, chars=len(text), content=text,
                                  status="irrelevant", fetched_at=now())
        self._read_pairs.add((source_id, q["id"]))

        chunk_size = max(2000, self.s.llm_context_chars - 2500)
        chunks = chunk_text(text, chunk_size, max_chunks=3)
        norm_page = _norm_text(text)
        saved = rejected = 0
        for part, chunk in enumerate(chunks, 1):
            try:
                data = self.ask_json(
                    f"Research question: {q['text']}\n(Overall topic: {self.topic['title']})\n\n"
                    f"Source: {title} ({url})\n---BEGIN SOURCE TEXT---\n{chunk}\n---END SOURCE TEXT---\n\n"
                    "Extract the facts from this source text that help answer the research question. Each note must "
                    "be a self-contained factual statement (include numbers, dates, names and context), and must be "
                    "accompanied by a short verbatim quote copied exactly from the source text that supports it. "
                    "Do not use outside knowledge. If the text is irrelevant, return an empty list.\n"
                    'Return JSON: {"notes": [{"claim": "...", "quote": "exact words from the text"}]}'
                )
            except LLMError as exc:
                self.log(f"Could not extract notes from part {part}/{len(chunks)} of {title or url}: {exc}", "warning")
                continue
            notes = data.get("notes", []) if isinstance(data, dict) else data
            if not isinstance(notes, list):
                continue
            for n in notes:
                if not isinstance(n, dict):
                    continue
                claim, quote = str(n.get("claim", "")).strip(), str(n.get("quote", "")).strip()
                if not claim:
                    continue
                if not quote_supported(quote, text, norm_page):
                    rejected += 1
                    continue
                self.db.add_note(self.topic_id, source_id, q["id"], claim, quote)
                saved += 1
        if saved:
            total = (known or {}).get("note_count", 0) + saved
            self.db.update_source(source_id, status="used", note_count=total)
            if known:
                known.update(status="used", note_count=total)
        msg = f"{'Re-read' if known else 'Read'} {title or url}: {saved} note(s)"
        if rejected:
            msg += f", {rejected} discarded (quote not found in source)"
        self.log(msg)
        return saved > 0

    def reflect(self, iteration: int) -> bool:
        questions = self.db.questions(self.topic_id)
        if iteration >= self.opts.max_iterations:
            return True
        if self.db.count_sources(self.topic_id, "used") >= self.opts.max_sources:
            return True
        summary = []
        for q in questions:
            claims = [n["claim"] for n in self.db.notes(self.topic_id, q["id"])]
            summary.append(f"Q: {q['text']} ({len(claims)} notes)\n" + "\n".join(f"  - {c[:200]}" for c in claims[:8]))
        digest = "\n".join(summary)[: self.s.llm_context_chars - 2000]
        remaining = self.opts.max_questions
        try:
            data = self.ask_json(
                f"Research topic:\n{self.subject}\n\nWhat has been found so far:\n{digest}\n\n"
                "Assess whether this is enough to write a thorough, well-sourced report. If important aspects are "
                f"missing, thin, or contradictory, list up to {remaining} new follow-up research questions that "
                "fill those gaps (do not repeat existing questions).\n"
                'Return JSON: {"sufficient": true|false, "reasoning": "...", "follow_up_questions": ["..."]}',
                mode="deep",
            )
        except LLMError as exc:
            self.log(f"Coverage review failed ({exc}); writing the report from what was found.", "warning")
            return True
        if not isinstance(data, dict):
            return True
        reasoning = str(data.get("reasoning", "")).strip()
        follow = _clean_list(data.get("follow_up_questions", []), remaining)
        existing = {_norm_text(q["text"]) for q in questions}
        follow = [f for f in follow if _norm_text(f) not in existing]
        sufficient = bool(data.get("sufficient")) and not (
            # Don't accept "sufficient" when nothing was found at all.
            not self.db.notes(self.topic_id)
        )
        if reasoning:
            self.log(f"Coverage review: {reasoning}")
        if sufficient or not follow:
            self.log("Research judged sufficient." if sufficient else "No further questions to pursue.")
            return True
        for f in follow:
            self.db.add_question(self.topic_id, f, iteration=iteration + 1)
        self.log("Follow-up questions:\n" + "\n".join(f"• {f}" for f in follow))
        return False

    # ---- report -------------------------------------------------------------------------
    def write_report(self) -> None:
        self.set_state(status="writing", phase="Planning the report structure", progress=80)
        all_notes = self.db.notes(self.topic_id)
        if not all_notes:
            raise RuntimeError("No usable information was gathered, so no report could be written. "
                               "Check the search provider and model settings, then retry.")
        sources = {s["id"]: s for s in self.db.sources(self.topic_id) if s["status"] == "used"}
        valid_ids = set(sources)
        self._notes_by_source: dict[int, list[dict]] = {}
        for n in all_notes:
            self._notes_by_source.setdefault(n["source_id"], []).append(n)
        self.verify_stats = {"checked": 0, "supported": 0, "recited": 0, "removed": 0, "flagged": 0, "unchecked": 0}

        outline = self.make_outline(all_notes)
        headings = [sec["heading"] for sec in outline]
        sections: list[tuple[str, str]] = []
        for i, sec in enumerate(outline):
            self.set_state(phase=f"Writing section {i + 1}/{len(outline)}: {sec['heading']}",
                           progress=82 + 13 * i / max(1, len(outline)))
            text = self.write_section(sec, headings, valid_ids)
            if text:
                sections.append((sec["heading"], text))
        if not sections:
            raise RuntimeError("The model could not produce any cited report sections.")

        self.set_state(phase="Writing summary", progress=96)
        summary = self.write_summary(sections, valid_ids)
        body = "\n\n".join(f"## {h}\n\n{t}" for h, t in sections)

        unanswered = [q["text"] for q in self.db.questions(self.topic_id) if q["note_count"] == 0]
        parts = [f"# {self.topic['title']}", ""]
        if summary:
            parts += ["## Summary", "", summary, ""]
        parts += [body, ""]
        cited_md, refs = rpt.renumber("\n".join(parts), sources)

        limitations = (
            "## Scope and limitations\n\n"
            f"This report was compiled automatically by a local AI model ({self.llm.model}) from "
            f"{len(refs)} web source(s) on {date.today().isoformat()}. Every statement is linked to the "
            "source it was drawn from; verify important facts against the cited sources."
        )
        vs = self.verify_stats
        if vs["checked"]:
            limitations += (
                f"\n\nThe model then checked each cited statement against the notes taken from its sources: "
                f"{vs['supported']} of {vs['checked']} were confirmed"
                + (f", {vs['recited']} were re-attributed to the source that supports them" if vs["recited"] else "")
                + (f", {vs['removed']} were removed as unsupported" if vs["removed"] else "")
                + (f", {vs['flagged']} are marked *[unverified]*" if vs["flagged"] else "")
                + (f", and {vs['unchecked']} could not be checked" if vs["unchecked"] else "")
                + ". This check is itself done by the model, so it reduces but does not eliminate errors."
            )
        if unanswered:
            limitations += "\n\nThe following questions could not be answered from the sources found:\n\n" + \
                "\n".join(f"- {u}" for u in unanswered)
        final_body = cited_md.rstrip() + "\n\n" + limitations
        report_md = final_body + "\n\n" + rpt.references_markdown(refs) + "\n"
        self.db.update_topic(self.topic_id, report_md=report_md, report_refs=refs, status="completed",
                             phase="Report ready", progress=100, completed_at=now())
        self.log(f"Report complete: {len(sections)} section(s), {len(refs)} cited source(s).")

    # -- outline --
    @staticmethod
    def _note_id(value: Any) -> int | None:
        m = re.fullmatch(r"\s*N?(\d+)\s*", str(value), re.IGNORECASE)
        return int(m.group(1)) if m else None

    def make_outline(self, notes: list[dict]) -> list[dict]:
        """Group the notes into a logical set of sections, merging overlapping questions.

        Returns [{"heading": str, "note_ids": [int, ...]}]. Falls back to one section per question
        when the model can't produce a usable outline."""
        questions = self.db.questions(self.topic_id)
        q_index = {q["id"]: i + 1 for i, q in enumerate(questions)}
        budget = self.s.llm_context_chars - 3000
        listed, lines, used = [], [], 0
        for n in notes:
            line = f"N{n['id']} (Q{q_index.get(n['question_id'], '?')}, S{n['source_id']}) {n['claim'][:220]}"
            if used + len(line) > budget:
                break
            lines.append(line)
            listed.append(n["id"])
            used += len(line) + 1
        q_lines = "\n".join(f"Q{q_index[q['id']]}: {q['text']}" for q in questions if q["note_count"])
        try:
            data = self.ask_json(
                f"You are planning the structure of a research report on:\n{self.subject}\n\n"
                f"Research questions:\n{q_lines}\n\nResearch notes (id, question, source, content):\n"
                + "\n".join(lines) + "\n\n"
                "Design a clear, logical outline of 3 to 7 sections that reads well for a human: for example "
                "background first, then the main findings, comparisons and open issues. Merge material that "
                "overlaps, even if it came from different questions; only keep one section per question if that "
                "really is the best structure. Give each section a short, specific heading. Assign each relevant "
                "note to exactly one section by its id, and leave out notes that are off-topic or repeat another.\n"
                'Return JSON: {"sections": [{"heading": "...", "notes": ["N1", "N4"]}]}',
                mode="deep",
            )
        except LLMError as exc:
            self.log(f"Could not plan the report outline ({exc}); using one section per question.", "warning")
            return self._outline_by_question(notes, questions)

        raw = data.get("sections") if isinstance(data, dict) else data
        valid = {n["id"] for n in notes}
        assigned: set[int] = set()
        outline = []
        for sec in raw if isinstance(raw, list) else []:
            if not isinstance(sec, dict):
                continue
            heading = re.sub(r"^#+\s*", "", str(sec.get("heading") or "")).strip()[:120]
            ids = []
            for v in sec.get("notes") or []:
                nid = self._note_id(v)
                if nid in valid and nid not in assigned:
                    ids.append(nid)
                    assigned.add(nid)
            if ids:
                outline.append({"heading": heading or f"Section {len(outline) + 1}", "note_ids": ids})
        outline = outline[:8]
        if not outline:
            self.log("The model returned no usable outline; using one section per question.", "warning")
            return self._outline_by_question(notes, questions)
        coverage = len(assigned & set(listed)) / max(1, len(listed))
        if coverage < 0.4:
            self.log(f"The outline used only {coverage:.0%} of the notes; using one section per question "
                     "so findings aren't lost.", "warning")
            return self._outline_by_question(notes, questions)

        # Notes the model never saw (context budget) join the section holding most of their question.
        by_id = {n["id"]: n for n in notes}
        unseen = [n for n in notes if n["id"] not in set(listed)]
        for n in unseen:
            best = max(outline, key=lambda sec: sum(by_id[i]["question_id"] == n["question_id"] for i in sec["note_ids"]))
            if any(by_id[i]["question_id"] == n["question_id"] for i in best["note_ids"]):
                best["note_ids"].append(n["id"])
                assigned.add(n["id"])
        left_out = len(valid) - len(assigned)
        self.log("Report outline:\n" + "\n".join(f"• {sec['heading']} ({len(sec['note_ids'])} notes)" for sec in outline)
                 + (f"\n{left_out} note(s) left out as off-topic or repetitive." if left_out else ""))
        return outline

    def _outline_by_question(self, notes: list[dict], questions: list[dict]) -> list[dict]:
        outline = []
        for q in questions:
            ids = [n["id"] for n in notes if n["question_id"] == q["id"]]
            if ids:
                outline.append({"heading": q["text"].rstrip("?"), "note_ids": ids})
        return outline

    # -- sections --
    def _notes_block(self, notes: list[dict], budget: int) -> str:
        lines, used = [], 0
        for n in notes:
            line = f"[S{n['source_id']}] {n['claim']}"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def write_section(self, sec: dict, headings: list[str], valid_ids: set[int]) -> str:
        by_id = {n["id"]: n for ns in self._notes_by_source.values() for n in ns}
        notes = [by_id[i] for i in sec["note_ids"] if i in by_id]
        notes_block = self._notes_block(notes, self.s.llm_context_chars - 3000)
        outline_txt = "\n".join(f"{i + 1}. {h}" + ("   <- this section" if h == sec["heading"] else "")
                                for i, h in enumerate(headings))
        prompt = (
            f"You are writing one section of a research report on: {self.topic['title']}\n\n"
            f"Report outline:\n{outline_txt}\n\n"
            f"Write the section \"{sec['heading']}\". The other sections cover their own topics, so stay on "
            "this one and don't repeat them.\n\n"
            f"Research notes for this section (each tagged with its source id):\n{notes_block}\n\n"
            "Write the body of the section in Markdown:\n"
            "- Do not repeat the section heading; it is added for you. Use ### subheadings only if needed.\n"
            "- 1-5 clear, well-organised paragraphs that synthesise the notes into a coherent explanation "
            "(bullet lists are fine for enumerations).\n"
            "- Use ONLY information from the notes above. Do not add outside knowledge.\n"
            "- EVERY sentence containing a fact must end with the source tag(s) it came from, exactly as given, "
            "e.g. \"... in 2023 [S4].\" or \"... [S2, S7].\"\n"
            "- Where sources disagree, say so and cite both.\n"
            "Output only the section body."
        )
        label = f"section '{sec['heading']}'"
        try:
            text = self._finalize_section(self.ask_text(prompt), label, valid_ids)
            if text is None:
                self.log(f"The {label} had no valid citations; retrying once.", "warning")
                text = self._finalize_section(self.ask_text(
                    prompt + "\n\nIMPORTANT: your previous attempt omitted the [S#] source tags. Every factual "
                    "sentence MUST end with one."), label, valid_ids)
        except LLMError as exc:
            self.log(f"Could not write the {label}: {exc}", "warning")
            return ""
        return text or ""

    def _finalize_section(self, text: str, label: str, valid_ids: set[int]) -> str | None:
        """Clean up a section body; returns None if nothing cited survives."""
        text = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", text.strip()).strip()
        lines = text.split("\n")
        if lines and re.match(r"#{1,2}\s", lines[0]):  # the model repeated the heading anyway
            lines = lines[1:]
        lines = ["### " + ln.lstrip("# ").strip() if ln.startswith("#") else ln for ln in lines]
        text = rpt.drop_invalid_citations("\n".join(lines).strip(), valid_ids)
        if not rpt.extract_source_ids(text):
            return None
        text, uncited = rpt.enforce_citations(text, self.s.uncited_policy)
        if uncited:
            verb = "removed" if self.s.uncited_policy == "drop" else "flagged"
            self.log(f"{uncited} uncited passage(s) {verb} from the {label}.", "warning")
        text = self.verify_citations(text, label)
        return text if rpt.extract_source_ids(text) else None

    # -- citation verification --
    def _evidence_block(self, source_ids: set[int], budget: int) -> str:
        """Notes from the cited sources, taken round-robin so every source is represented."""
        queues = {sid: list(self._notes_by_source.get(sid, [])) for sid in sorted(source_ids)}
        lines, used = [], 0
        while any(queues.values()):
            for sid, q in queues.items():
                if not q:
                    continue
                n = q.pop(0)
                line = f"[S{sid}] {n['claim']} (quote: \"{n['quote']}\")"
                if used + len(line) > budget:
                    queues = {k: [] for k in queues}
                    break
                lines.append(line)
                used += len(line) + 1
        return "\n".join(lines)

    def verify_citations(self, text: str, label: str) -> str:
        """Check each cited statement against the notes of the sources it cites.

        Supported statements stay; statements backed by a different source are re-cited; the rest
        are removed (or flagged with UNCITED_POLICY=flag). If the check can't run, the text is
        kept as written and counted as unchecked."""
        if not self.s.verify_citations:
            return text
        units = rpt.citation_units(text)
        if not units:
            return text
        stats = self.verify_stats
        statements = "\n".join(f"{u.id}. {u.statement} (cites: {', '.join(f'S{i}' for i in u.source_ids)})"
                               for u in units)
        cited = {i for u in units for i in u.source_ids}
        evidence = self._evidence_block(cited, max(2000, self.s.llm_context_chars - 2500 - len(statements)))
        try:
            data = self.ask_json(
                "Check each numbered statement from a research report against the evidence. The evidence lists "
                "notes taken from each source, tagged with the source id, each with a verbatim quote from the "
                "source. Judge only against this evidence, never your own knowledge.\n\n"
                f"Evidence:\n{evidence}\n\nStatements:\n{statements}\n\n"
                "For each statement give a verdict:\n"
                "- \"supported\": the evidence from the cited source(s) states it, or it is a faithful "
                "paraphrase or summary of it.\n"
                "- \"wrong_source\": the evidence supports it, but under a different source id than the one "
                "cited. Put the correct id(s) in \"sources\".\n"
                "- \"unsupported\": the cited evidence does not back it up, or the statement adds facts, numbers, "
                "dates, causes or certainty that the evidence does not contain.\n"
                'Return JSON: {"results": [{"id": 1, "verdict": "supported", "sources": [], "reason": "short"}]}'
            )
        except LLMError as exc:
            stats["checked"] += len(units)
            stats["unchecked"] += len(units)
            self.log(f"Could not check the citations in the {label} ({exc}); kept as written.", "warning")
            return text

        results: dict[int, dict] = {}
        raw = data.get("results") if isinstance(data, dict) else data
        for r in raw if isinstance(raw, list) else []:
            if isinstance(r, dict) and self._note_id(r.get("id")) is not None:
                results[self._note_id(r.get("id"))] = r
        evidence_ids = {int(m) for m in re.findall(r"^\[S(\d+)\]", evidence, re.M)}
        removed: list[str] = []
        before = dict(stats)

        def decide(u: rpt.Unit) -> str:
            stats["checked"] += 1
            r = results.get(u.id)
            if r is None:
                stats["unchecked"] += 1
                return u.text
            verdict = str(r.get("verdict", "")).strip().lower()
            if verdict == "supported":
                stats["supported"] += 1
                return u.text
            if verdict == "wrong_source":
                ids = sorted({int(x) for x in re.findall(r"\d+", " ".join(map(str, r.get("sources") or [])))}
                             & evidence_ids)
                if ids:
                    stats["recited"] += 1
                    return rpt.recite(u.text, ids)
            reason = str(r.get("reason", "")).strip()
            removed.append(f"“{u.statement[:160]}”" + (f" ({reason[:120]})" if reason else ""))
            if self.s.uncited_policy == "flag":
                stats["flagged"] += 1
                return u.text.rstrip() + " *[unverified]*"
            stats["removed"] += 1
            return ""

        text = rpt.rewrite_units(text, decide)
        delta = {k: stats[k] - before[k] for k in stats}
        msg = (f"Citation check, {label}: {delta['supported']}/{delta['checked']} confirmed"
               + (f", {delta['recited']} re-attributed" if delta["recited"] else "")
               + (f", {delta['removed']} removed" if delta["removed"] else "")
               + (f", {delta['flagged']} flagged" if delta["flagged"] else "")
               + (f", {delta['unchecked']} not checked" if delta["unchecked"] else ""))
        if removed:
            msg += "\nUnsupported:\n" + "\n".join(f"• {r}" for r in removed)
        self.log(msg, "warning" if removed else "info")
        return text

    # -- summary --
    def write_summary(self, sections: list[tuple[str, str]], valid_ids: set[int]) -> str:
        # Give every section a fair share of the context instead of truncating the end away.
        share = max(600, (self.s.llm_context_chars - 2500) // max(1, len(sections)))
        condensed = "\n\n".join(f"## {h}\n{t[:share]}" for h, t in sections)
        try:
            text = self.ask_text(
                f"Below is the body of a research report on: {self.subject}\n\n---\n{condensed}\n---\n\n"
                "Write an executive summary of 1-3 short paragraphs (or a short bullet list of key findings) that "
                "directly answers the topic, drawing on all sections. Use ONLY facts stated in the report body and "
                "keep their source tags (e.g. [S4]) at the end of each sentence. Do not add a heading. Output only "
                "the summary."
            )
        except LLMError as exc:
            self.log(f"Could not write the summary: {exc}", "warning")
            return ""
        text = rpt.drop_invalid_citations(text.strip(), valid_ids)
        text, _ = rpt.enforce_citations(text, "drop")
        return self.verify_citations(text, "summary") if text else ""


def build_agent(topic_id: int, db: Database, settings: Settings, llm: LLMClient,
                cancel_event: threading.Event | None = None,
                searcher_factory: Callable[[Settings], Searcher] = Searcher,
                fetcher_factory: Callable[[Settings], Fetcher] = Fetcher) -> ResearchAgent:
    topic = db.get_topic(topic_id)
    model = (topic or {}).get("model") or settings.llm_model
    return ResearchAgent(topic_id, db, llm.with_model(model), searcher_factory(settings),
                         fetcher_factory(settings), settings, cancel_event)
