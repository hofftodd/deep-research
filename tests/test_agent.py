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
        "research questions that together", "Assess whether", "executive summary"))
    agent.run()
    t = db.get_topic(tid)
    assert t["status"] == "completed"
    log = "\n".join(messages(db, tid))
    assert "Planning failed" in log and "Could not write the summary" in log
    # Planning fell back to the title as the only question; its section was still written.
    assert [q["text"] for q in db.questions(tid)] == ["Sodium-ion batteries for grid storage"]
    assert "## Summary" not in t["report_md"] and "## How sodium-ion cells work" in t["report_md"]


def test_failed_section_is_left_out(tmp_path):
    db, tid, agent = make(tmp_path, FlakyLLM('Write the section "Costs and deployment"'))
    agent.run()
    t = db.get_topic(tid)
    assert t["status"] == "completed"
    assert "## How sodium-ion cells work" in t["report_md"] and "## Costs and deployment" not in t["report_md"]
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


# ---- report outline (section merging) -----------------------------------------------------------
def test_outline_merges_questions_into_themed_sections(tmp_path):
    llm = FakeLLM()
    db, tid, agent = make(tmp_path, llm)
    agent.run()
    md = db.get_topic(tid)["report_md"]
    # Sections follow the model's outline (heading markup stripped), not the research questions.
    assert "## How sodium-ion cells work" in md and "## Costs and deployment" in md
    assert "What are sodium-ion batteries" not in md.split("## Scope")[0]

    outline = agent.make_outline(db.notes(tid))
    ids = [i for sec in outline for i in sec["note_ids"]]
    assert len(ids) == len(set(ids))                        # each note in one section only
    assert set(ids) <= {n["id"] for n in db.notes(tid)}     # invented id N424242 ignored

    section_prompts = [p for p in llm.prompts if "one section of a research report" in p]
    assert len(section_prompts) == 2
    assert "1. How sodium-ion cells work   <- this section" in section_prompts[0]
    assert "2. Costs and deployment" in section_prompts[0]   # writer sees the whole outline


def test_unusable_outline_falls_back_to_one_section_per_question(tmp_path):
    db, tid, agent = make(tmp_path, FakeLLM(outline=False))
    agent.run()
    md = db.get_topic(tid)["report_md"]
    assert "## What are sodium-ion batteries" in md and "## How do they perform for grid storage" in md
    assert any("no usable outline" in e["message"] for e in db.events(tid))


def test_failed_outline_call_falls_back(tmp_path):
    db, tid, agent = make(tmp_path, FlakyLLM("planning the structure"))
    agent.run()
    assert db.get_topic(tid)["status"] == "completed"
    assert any("Could not plan the report outline" in e["message"] for e in db.events(tid))


def test_notes_beyond_the_outline_budget_join_their_question_section(tmp_path):
    db, tid, agent = make(tmp_path)
    agent.run()
    notes = db.notes(tid)
    agent.s.llm_context_chars = 3000 + 90  # room to list only the first note or so
    outline = agent.make_outline(notes)
    assert sorted(i for sec in outline for i in sec["note_ids"]) == sorted(n["id"] for n in notes)


def test_summary_sees_every_section(tmp_path):
    llm = FakeLLM()
    db, tid, agent = make(tmp_path, llm)
    agent.run()
    summary_prompt = next(p for p in llm.prompts if "executive summary" in p)
    assert "## How sodium-ion cells work" in summary_prompt and "## Costs and deployment" in summary_prompt


# ---- citation verification ----------------------------------------------------------------------
def test_unsupported_statements_are_removed_and_reported(tmp_path):
    db, tid, agent = make(tmp_path, FakeLLM(unsupported=("favour",)))
    agent.run()
    t = db.get_topic(tid)
    body = t["report_md"].split("## Scope")[0]
    assert "favour" not in body and "abundant and cheap" in body
    assert "were removed as unsupported" in t["report_md"]
    log = "\n".join(e["message"] for e in db.events(tid))
    assert "Unsupported:\n• “Grid projects favour cost.” (not in evidence)" in log


def test_unsupported_statements_flagged_with_flag_policy(tmp_path):
    db, tid, agent = make(tmp_path, FakeLLM(unsupported=("favour",)))
    agent.s.uncited_policy = "flag"
    agent.run()
    md = db.get_topic(tid)["report_md"]
    assert md.count("Grid projects favour cost [1, 2]. *[unverified]*") == 2
    assert "are marked *[unverified]*" in md


def _verifier(tmp_path, llm):
    db, tid, agent = make(tmp_path, llm)
    agent._notes_by_source = {
        1: [{"claim": "Sodium is plentiful", "quote": "sodium is plentiful in seawater"}],
        2: [{"claim": "Sodium is abundant", "quote": "sodium is abundant in the crust"}],
    }
    agent.verify_stats = dict.fromkeys(("checked", "supported", "recited", "removed", "flagged", "unchecked"), 0)
    return db, tid, agent


def test_misattributed_statement_is_recited_to_the_supporting_source(tmp_path):
    _, _, agent = _verifier(tmp_path, FakeLLM(wrong_source=("abundant", 2)))
    out = agent.verify_citations("Sodium is abundant [S1]. It is plentiful [S1, S2].", "test")
    assert out == "Sodium is abundant [S2]. It is plentiful [S1, S2]."
    assert agent.verify_stats["recited"] == 1 and agent.verify_stats["supported"] == 1


def test_recite_to_a_source_outside_the_evidence_removes_instead(tmp_path):
    _, _, agent = _verifier(tmp_path, FakeLLM(wrong_source=("abundant", 99)))
    out = agent.verify_citations("Sodium is abundant [S1]. It is plentiful [S2].", "test")
    assert out == "It is plentiful [S2]." and agent.verify_stats["removed"] == 1


def test_verifier_sees_quotes_from_every_cited_source(tmp_path):
    llm = FakeLLM()
    _, _, agent = _verifier(tmp_path, llm)
    agent.verify_citations("A [S1]. B [S2].", "test")
    prompt = llm.prompts[-1]
    assert '[S1] Sodium is plentiful (quote: "sodium is plentiful in seawater")' in prompt
    assert "1. A. (cites: S1)" in prompt and "2. B. (cites: S2)" in prompt


def test_missing_verdicts_and_failed_checks_keep_text(tmp_path):
    db, tid, agent = _verifier(tmp_path, FlakyLLM("Check each numbered statement"))
    out = agent.verify_citations("A [S1]. B [S2].", "test")
    assert out == "A [S1]. B [S2]." and agent.verify_stats["unchecked"] == 2
    assert any("Could not check the citations" in e["message"] for e in db.events(tid))

    class PartialLLM(FakeLLM):
        def chat_json(self, system, user, mode="fast", retries=2):
            return {"results": [{"id": "1", "verdict": "supported"}]}  # says nothing about #2

    _, _, agent = _verifier(tmp_path, PartialLLM())
    assert agent.verify_citations("A [S1]. B [S2].", "test") == "A [S1]. B [S2]."
    assert agent.verify_stats["unchecked"] == 1 and agent.verify_stats["supported"] == 1


def test_verification_can_be_turned_off(tmp_path):
    llm = FakeLLM(unsupported=("favour",))
    db, tid, agent = make(tmp_path, llm)
    agent.s.verify_citations = False
    agent.run()
    assert not any("Check each numbered statement" in p for p in llm.prompts)
    assert "favour" in db.get_topic(tid)["report_md"]


def test_outline_that_drops_most_notes_falls_back(tmp_path):
    class StingyLLM(FakeLLM):
        def chat_json(self, system, user, mode="fast", retries=2):
            if "planning the structure" in user:
                first = re.search(r"^N(\d+) ", user, re.M).group(1)
                return {"sections": [{"heading": "Just one note", "notes": [f"N{first}"]}]}
            return super().chat_json(system, user, mode=mode)

    db, tid, agent = make(tmp_path, StingyLLM())
    agent.run()
    md = db.get_topic(tid)["report_md"]
    assert "Just one note" not in md and "## What are sodium-ion batteries" in md
    assert any("used only" in e["message"] for e in db.events(tid))
