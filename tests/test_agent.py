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
