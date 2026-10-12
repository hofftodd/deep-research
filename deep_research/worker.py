"""Background worker that runs queued research topics one (or N) at a time."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from .agent import ResearchAgent, build_agent
from .config import Settings
from .db import Database
from .llm import LLMClient

log = logging.getLogger(__name__)

AgentFactory = Callable[[int, threading.Event], ResearchAgent]


class Worker:
    def __init__(self, db: Database, settings: Settings, llm: LLMClient,
                 agent_factory: AgentFactory | None = None):
        self.db = db
        self.settings = settings
        self.llm = llm
        self._factory = agent_factory or (lambda tid, ev: build_agent(tid, db, settings, llm, ev))
        self._queue: "queue.Queue[int | None]" = queue.Queue()
        self._cancel: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        interrupted = self.db.mark_interrupted()
        if interrupted:
            log.warning("Marked %d interrupted topic(s) as failed", interrupted)
        for _ in range(max(1, self.settings.worker_concurrency)):
            t = threading.Thread(target=self._loop, daemon=True, name="research-worker")
            t.start()
            self._threads.append(t)
        for tid in self.db.queued_topic_ids():
            self.enqueue(tid)

    def stop(self) -> None:
        with self._lock:
            for ev in self._cancel.values():
                ev.set()
        for _ in self._threads:
            self._queue.put(None)

    def enqueue(self, topic_id: int) -> None:
        with self._lock:
            self._cancel[topic_id] = threading.Event()
        self._queue.put(topic_id)

    def cancel(self, topic_id: int) -> bool:
        """Request cancellation. Returns True if the topic was queued or running."""
        with self._lock:
            ev = self._cancel.get(topic_id)
        if ev is None:
            return False
        ev.set()
        topic = self.db.get_topic(topic_id)
        if topic and topic["status"] == "queued":
            self.db.update_topic(topic_id, status="cancelled", phase="Cancelled before starting")
            self.db.log(topic_id, "Cancelled before research started.", "warning")
        return True

    def _loop(self) -> None:
        while True:
            topic_id = self._queue.get()
            if topic_id is None:
                return
            with self._lock:
                ev = self._cancel.get(topic_id)
            topic = self.db.get_topic(topic_id)
            if ev is None or ev.is_set() or not topic or topic["status"] != "queued":
                continue
            try:
                self._factory(topic_id, ev).run()
            except Exception as exc:  # noqa: BLE001 - the agent normally records the failure itself
                log.exception("Research for topic %s failed", topic_id)
                current = self.db.get_topic(topic_id)
                if current and current["status"] == "queued":  # failed before the agent started
                    self.db.update_topic(topic_id, status="failed", error=str(exc))
            finally:
                with self._lock:
                    if self._cancel.get(topic_id) is ev:
                        del self._cancel[topic_id]
