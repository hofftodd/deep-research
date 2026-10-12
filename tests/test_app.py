import time

from fastapi.testclient import TestClient

from deep_research.agent import build_agent
from deep_research.app import create_app
from deep_research.config import Settings
from deep_research.db import Database
from deep_research.worker import Worker
from tests.fakes import FakeFetcher, FakeLLM, FakeSearcher


def make_client(tmp_path):
    settings = Settings(data_dir=tmp_path)
    db = Database(tmp_path / "t.db")
    llm = FakeLLM()
    worker = Worker(db, settings, llm, agent_factory=lambda tid, ev: build_agent(
        tid, db, settings, llm, ev, searcher_factory=FakeSearcher, fetcher_factory=FakeFetcher))
    return TestClient(create_app(settings, worker=worker, db=db, llm=llm))


def wait_for(client, tid, statuses, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = client.get(f"/api/topics/{tid}").json()
        if t["status"] in statuses:
            return t
        time.sleep(0.05)
    raise AssertionError(f"topic stuck in {t['status']}")


def test_create_run_and_download(tmp_path):
    with make_client(tmp_path) as client:
        assert client.get("/").status_code == 200
        health = client.get("/api/health").json()
        assert health["llm_ok"] and "fake-model" in health["models"]

        r = client.post("/api/topics", json={"title": "Sodium-ion batteries", "max_iterations": 1})
        assert r.status_code == 201
        tid = r.json()["id"]
        t = wait_for(client, tid, {"completed", "failed"})
        assert t["status"] == "completed", t["error"]
        assert t["report_html"] and 'class="references"' in t["report_html"]
        assert t["questions"] and t["sources"]

        assert client.get(f"/api/topics/{tid}/events").json()
        md = client.get(f"/api/topics/{tid}/report.md")
        assert md.status_code == 200 and "## References" in md.text
        assert "attachment" in md.headers["content-disposition"]

        listing = client.get("/api/topics").json()
        assert listing[0]["id"] == tid and listing[0]["source_count"] > 0

        assert client.post(f"/api/topics/{tid}/retry").status_code == 200
        wait_for(client, tid, {"completed"})
        assert client.delete(f"/api/topics/{tid}").status_code == 200
        assert client.get(f"/api/topics/{tid}").status_code == 404


def test_validation(tmp_path):
    with make_client(tmp_path) as client:
        assert client.post("/api/topics", json={"title": "x"}).status_code == 422
        assert client.post("/api/topics/999/cancel").status_code == 404
