import re

import pytest

from deep_research.agent import ResearchAgent
from deep_research.config import Settings
from deep_research.db import Database
from tests.fakes import FakeFetcher, FakeLLM, FakeSearcher


def make(tmp_path, llm=None, **opts):
    settings = Settings(data_dir=tmp_path)
    db = Database(":memory:")
    tid = db.create_topic("Sodium-ion batteries for grid storage", "Focus on 2023 onwards", "fake-model", opts)
    agent = ResearchAgent(tid, db, llm or FakeLLM(), FakeSearcher(), FakeFetcher(), settings)
    return db, tid, agent


def test_full_run_produces_fully_cited_report(tmp_path):
    db, tid, agent = make(tmp_path, FakeLLM(cite_bogus=True, uncited_paragraph=True))
    agent.run()
    t = db.get_topic(tid)
    assert t["status"] == "completed" and t["progress"] == 100
    md = t["report_md"]
    assert md.startswith("# Sodium-ion batteries for grid storage")
    assert "## Summary" in md and "## References" in md

    body = md.split("## Scope and limitations")[0]
    # The invented citation and the uncited paragraph are gone.
    assert "invented fact" not in body and "no citation at all" not in body and "made-up source" not in body
    # Every non-heading paragraph in the body carries a citation.
    for block in re.split(r"\n\s*\n", body.strip()):
        if not block.startswith("#"):
            assert re.search(r"\[\d+(, \d+)*\]", block), block

    cited = {int(n) for m in re.findall(r"\[(\d+(?:, \d+)*)\]", body) for n in m.split(", ")}
    refs = t["report_refs"]
    assert cited == {r["n"] for r in refs}
    used = {s["id"] for s in db.sources(tid) if s["status"] == "used"}
    assert {r["source_id"] for r in refs} <= used


def test_fabricated_quotes_are_rejected(tmp_path):
    db, tid, agent = make(tmp_path)
    agent.run()
    claims = [n["claim"] for n in db.notes(tid)]
    assert claims and not any("fabricated" in c for c in claims)
    assert any("discarded" in e["message"] for e in db.events(tid))


def test_failed_fetch_is_recorded(tmp_path):
    db, tid, agent = make(tmp_path, sources_per_question=5)
    agent.run()
    broken = [s for s in db.sources(tid) if "broken" in s["url"]]
    assert broken and broken[0]["status"] == "failed"


def test_cancel(tmp_path):
    db, tid, agent = make(tmp_path)
    agent.cancel_event.set()
    agent.run()
    assert db.get_topic(tid)["status"] == "cancelled"


def test_no_information_fails_cleanly(tmp_path):
    db, tid, agent = make(tmp_path)
    agent.searcher.search = lambda q, limit=None: []
    with pytest.raises(RuntimeError):
        agent.run()
    t = db.get_topic(tid)
    assert t["status"] == "failed" and "No usable information" in t["error"]


def test_sources_are_reused_across_questions(tmp_path):
    db, tid, agent = make(tmp_path)
    agent.run()
    qs = db.questions(tid)
    assert all(q["status"] == "answered" for q in qs)
    # Three pages, each mined for both questions, fetched only once.
    used = [s for s in db.sources(tid) if s["status"] == "used"]
    assert len(used) == 3 and all(s["note_count"] == 2 for s in used)


# ---- resilience: a failed model call degrades the run instead of killing it ----------------------
from deep_research.llm import LLMError  # noqa: E402


class FlakyLLM(FakeLLM):
    """Fails every call whose prompt contains one of the given markers."""

    def __init__(self, *markers, warn=None):
        super().__init__()
        self.markers = markers
        self.warn = warn

    def _maybe_fail(self, prompt):
        if self.warn:
            self.warnings.append(self.warn)
            self.warn = None
        if any(m in prompt for m in self.markers):
            raise LLMError("simulated failure")

    def chat_json(self, system, user, mode="fast", retries=2):
        self._maybe_fail(user)
        return super().chat_json(system, user, mode=mode)

    def chat(self, messages, json_mode=False, mode="fast"):
        self._maybe_fail(messages[-1]["content"])
        return super().chat(messages, json_mode=json_mode, mode=mode)


def messages(db, tid):
    return [e["message"] for e in db.events(tid)]


def test_failed_note_extraction_skips_that_page_only(tmp_path):
    db, tid, agent = make(tmp_path, FlakyLLM("Source: Battery basics"))
    agent.run()
    assert db.get_topic(tid)["status"] == "completed"
    assert any("Could not extract notes" in m and "Battery basics" in m for m in messages(db, tid))
    assert {s["url"] for s in db.sources(tid) if s["status"] == "used"} == {
        "https://news.example.net/sodium", "https://example.com/grid-storage"}


def test_failed_planning_review_section_and_summary_degrade_gracefully(tmp_path):
    db, tid, agent = make(tmp_path, FlakyLLM(
        "research questions that together", "Assess whether", "executive summary", "This section answers: How do"))
    agent.run()
    t = db.get_topic(tid)
    assert t["status"] == "completed"
    log = "\n".join(messages(db, tid))
    assert "Planning failed" in log and "Could not write the summary" in log
    # Planning fell back to the title as the only question; its section was still written.
    assert [q["text"] for q in db.questions(tid)] == ["Sodium-ion batteries for grid storage"]
    assert "## Summary" not in t["report_md"] and "## Findings" in t["report_md"]


def test_failed_section_is_left_out(tmp_path):
    db, tid, agent = make(tmp_path, FlakyLLM("This section answers: How do"))
    agent.run()
    t = db.get_topic(tid)
    assert t["status"] == "completed" and t["report_md"].count("## Findings") == 1
    assert any("Could not write the section" in m for m in messages(db, tid))


def test_model_down_stops_after_consecutive_failures(tmp_path):
    llm = FlakyLLM("")  # every call fails
    db, tid, agent = make(tmp_path, llm)
    with pytest.raises(Exception, match="3 model calls failed in a row"):
        agent.run()
    t = db.get_topic(tid)
    assert t["status"] == "failed" and "failed in a row" in t["error"]
    assert llm.calls == 0  # FlakyLLM raises before counting; no work was attempted past the limit


def test_one_success_resets_the_failure_count(tmp_path):
    # Two pages fail (two failures), then a success; with the limit at 3 the run continues.
    db, tid, agent = make(tmp_path, FlakyLLM("Source: Sodium-ion news", "Source: Battery basics"))
    agent.run()
    assert db.get_topic(tid)["status"] == "completed"


def test_calls_use_the_right_mode(tmp_path):
    llm = FakeLLM()
    db, tid, agent = make(tmp_path, llm)
    agent.run()
    assert llm.modes[0] == "deep"  # planning
    assert llm.modes.count("deep") >= 3 and llm.modes.count("fast") >= 5


def test_client_warnings_and_usage_reach_the_activity_log(tmp_path):
    db, tid, agent = make(tmp_path, FlakyLLM(warn="The model is still thinking"))
    agent.run()
    log = messages(db, tid)
    assert "The model is still thinking" in log
    assert any(m.startswith("Model usage:") for m in log)
