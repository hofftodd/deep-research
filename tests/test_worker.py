import threading

from deep_research.config import Settings
from deep_research.db import Database
from deep_research.worker import Worker


class BlockingAgent:
    def __init__(self, started, release):
        self.started, self.release = started, release

    def run(self):
        self.started.set()
        self.release.wait(5)


def test_restart_marks_interrupted_runs_failed_and_resumes_queue(tmp_path):
    db = Database(":memory:")
    running = db.create_topic("was running", "", "m", {})
    db.update_topic(running, status="researching")
    queued = db.create_topic("was queued", "", "m", {})
    ran = []
    done = threading.Event()

    def factory(tid, ev):
        ran.append(tid)
        done.set()
        return BlockingAgent(threading.Event(), threading.Event())

    w = Worker(db, Settings(data_dir=tmp_path), llm=None, agent_factory=factory)
    w.start()
    assert done.wait(5)
    w.stop()
    t = db.get_topic(running)
    assert t["status"] == "failed" and "restart" in t["error"]
    assert ran == [queued]


def test_cancel_queued_topic_never_runs(tmp_path):
    db = Database(":memory:")
    started, release = threading.Event(), threading.Event()
    ran = []

    def factory(tid, ev):
        ran.append(tid)
        return BlockingAgent(started, release)

    w = Worker(db, Settings(data_dir=tmp_path), llm=None, agent_factory=factory)
    w.start()
    first = db.create_topic("first", "", "m", {})
    second = db.create_topic("second", "", "m", {})
    w.enqueue(first)
    w.enqueue(second)
    assert started.wait(5)  # first is running, second waits in the queue
    assert w.cancel(second)
    assert db.get_topic(second)["status"] == "cancelled"
    release.set()
    w.stop()
    for t in w._threads:
        t.join(5)
    assert ran == [first]
    assert not w.cancel(999)


def test_factory_error_marks_topic_failed(tmp_path):
    db = Database(":memory:")
    done = threading.Event()

    def factory(tid, ev):
        done.set()
        raise RuntimeError("bad setup")

    w = Worker(db, Settings(data_dir=tmp_path), llm=None, agent_factory=factory)
    w.start()
    tid = db.create_topic("t", "", "m", {})
    w.enqueue(tid)
    assert done.wait(5)
    w.stop()
    for t in w._threads:
        t.join(5)
    t = db.get_topic(tid)
    assert t["status"] == "failed" and t["error"] == "bad setup"
