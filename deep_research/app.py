"""FastAPI app: JSON API + the single-page web UI."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import report as rpt
from .config import Settings
from .db import ACTIVE_STATUSES, Database
from .llm import LLMClient
from .worker import Worker

STATIC = Path(__file__).parent / "static"


class TopicIn(BaseModel):
    title: str = Field(min_length=3, max_length=500)
    description: str = Field(default="", max_length=5000)
    model: str = ""
    max_iterations: int | None = Field(default=None, ge=1, le=10)
    max_questions: int | None = Field(default=None, ge=1, le=12)
    max_sources: int | None = Field(default=None, ge=1, le=100)
    sources_per_question: int | None = Field(default=None, ge=1, le=10)


def create_app(settings: Settings | None = None, worker: Worker | None = None,
               db: Database | None = None, llm: LLMClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    db = db or Database(settings.db_path)
    llm = llm or LLMClient(settings.llm_base_url, settings.llm_model, settings.llm_api_key,
                           settings.llm_timeout, profiles=settings.call_profiles(),
                           think_control=settings.llm_think_control)
    worker = worker or Worker(db, settings, llm)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="Deep Research", lifespan=lifespan)
    app.state.db, app.state.worker, app.state.settings = db, worker, settings

    def get_or_404(topic_id: int) -> dict:
        topic = db.get_topic(topic_id)
        if not topic:
            raise HTTPException(404, "Topic not found")
        return topic

    @app.get("/api/health")
    def health():
        out = {
            "model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "search_provider": settings.search_provider,
            "preset": settings.llm_preset,
            "thinking": {"fast": settings.llm_think_fast, "deep": settings.llm_think_deep},
            "llm_ok": False,
            "models": [],
            "defaults": {
                "max_iterations": settings.max_iterations,
                "max_questions": settings.max_questions,
                "max_sources": settings.max_sources,
                "sources_per_question": settings.sources_per_question,
            },
        }
        try:
            out["models"] = llm.list_models()
            out["llm_ok"] = True
        except Exception as exc:  # noqa: BLE001
            out["llm_error"] = str(exc)
        return out

    @app.get("/api/topics")
    def list_topics():
        return db.list_topics()

    @app.post("/api/topics", status_code=201)
    def create_topic(body: TopicIn):
        options = body.model_dump(include={"max_iterations", "max_questions", "max_sources", "sources_per_question"},
                                  exclude_none=True)
        tid = db.create_topic(body.title.strip(), body.description.strip(),
                              body.model.strip() or settings.llm_model, options)
        db.log(tid, "Topic created and queued.")
        worker.enqueue(tid)
        return db.get_topic(tid)

    @app.get("/api/topics/{topic_id}")
    def get_topic(topic_id: int):
        topic = get_or_404(topic_id)
        topic["questions"] = db.questions(topic_id)
        topic["sources"] = db.sources(topic_id)
        topic["note_count"] = len(db.notes(topic_id))
        body = (topic.get("report_md") or "").split("\n## References\n")[0]
        topic["report_html"] = rpt.render_html(body, topic["report_refs"]) if topic.get("report_md") else None
        return topic

    @app.get("/api/topics/{topic_id}/events")
    def get_events(topic_id: int, after: int = 0):
        get_or_404(topic_id)
        return db.events(topic_id, after)

    @app.get("/api/topics/{topic_id}/notes")
    def get_notes(topic_id: int):
        get_or_404(topic_id)
        return db.notes(topic_id)

    @app.get("/api/topics/{topic_id}/report.md")
    def report_markdown(topic_id: int):
        topic = get_or_404(topic_id)
        if not topic.get("report_md"):
            raise HTTPException(404, "Report not ready")
        slug = re.sub(r"[^a-z0-9]+", "-", topic["title"].lower()).strip("-")[:60] or "report"
        return PlainTextResponse(topic["report_md"], media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{slug}.md"'})

    @app.post("/api/topics/{topic_id}/cancel")
    def cancel(topic_id: int):
        topic = get_or_404(topic_id)
        if topic["status"] not in ("queued", *ACTIVE_STATUSES):
            raise HTTPException(409, "Topic is not running")
        if not worker.cancel(topic_id):
            db.update_topic(topic_id, status="cancelled", phase="Cancelled")
        db.log(topic_id, "Cancellation requested.", "warning")
        return {"ok": True}

    @app.post("/api/topics/{topic_id}/retry")
    def retry(topic_id: int):
        topic = get_or_404(topic_id)
        if topic["status"] in ("queued", *ACTIVE_STATUSES):
            raise HTTPException(409, "Topic is already running")
        db.reset_topic(topic_id)
        db.log(topic_id, "Research restarted.")
        worker.enqueue(topic_id)
        return db.get_topic(topic_id)

    @app.delete("/api/topics/{topic_id}")
    def delete(topic_id: int):
        topic = get_or_404(topic_id)
        if topic["status"] in ("queued", *ACTIVE_STATUSES):
            worker.cancel(topic_id)
        db.delete_topic(topic_id)
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app
