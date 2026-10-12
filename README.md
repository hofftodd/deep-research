# Deep Research

A self-hosted deep research tool. You give it a topic in a web page; an agent driven by a **local AI
model** plans research questions, searches the web, reads sources, takes notes, decides whether it
knows enough, and writes a human-readable report in which **every statement cites its source**.

## Features

- **Web dashboard**: create topics, see past topics, watch live status (phase, progress, activity
  log, research plan, sources examined), cancel / retry / delete, read the report, and download it as Markdown.
- **Runs on your machine**: any OpenAI-compatible model server works: Ollama, LM Studio,
  llama.cpp `llama-server`, vLLM, LocalAI. Nothing is sent to a hosted AI service.
- **Iterative agent**: plan → search → rank results → read pages → extract notes → review
  coverage → follow-up questions, repeated until coverage is sufficient or the limits you set
  (rounds, sources) are reached.
- **Citations are enforced by code, not just requested from the model** (see below).

## Quick start

Requires Python 3.10+.

```bash
# 1. A local model. Example with Ollama (https://ollama.com):
ollama pull qwen3:8b
OLLAMA_CONTEXT_LENGTH=16384 ollama serve     # a larger context window helps a lot

# 2. The app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                          # adjust model / search settings
python -m deep_research                       # http://127.0.0.1:8000
```

Open http://127.0.0.1:8000, click **New research**, and describe the topic. The header shows
whether the model server is reachable. You can pick any model the server offers per topic, and
override research depth under **Model and depth**.

### Web search

| `SEARCH_PROVIDER` | Setup | Notes |
|---|---|---|
| `duckduckgo` (default) | none | Convenient, but DuckDuckGo rate-limits automated queries and blocks some networks. |
| `searxng` (recommended) | `docker compose up -d` | Private metasearch on `localhost:8888`, uses the bundled `searxng/settings.yml` (JSON API enabled). Set `SEARXNG_URL=http://localhost:8888`. |
| `brave` | `BRAVE_API_KEY` | Brave Search API (free tier available). |

### Choosing a model

Research quality depends heavily on the model. It must follow instructions and produce JSON
reliably. 7–14B instruction-tuned models (Qwen3 8B/14B, Llama 3.1 8B, Gemma 3 12B, Mistral
Small) work; bigger ones write noticeably better reports. Reasoning models are fine because
`<think>` blocks are stripped. `LLM_CONTEXT_CHARS` caps how much text goes into one call; budget
about 3 characters per token of context and leave room for the reply. The default 12000 suits an
8k-token context or larger; raise it (e.g. 30000 with a 16k context) to read more of each page.

## How it works

```
topic ──► plan: 3–5 research questions
            │
            ▼
  ┌──► for each open question:
  │      model writes search queries ─► web search ─► model ranks results
  │      fetch page ─► extract text ─► model extracts notes + verbatim quotes
  │      (notes whose quote is not actually in the page are discarded)
  │    review coverage: sufficient?  ── no ─► follow-up questions ──┘
  │            │ yes / limits reached
  ▼            ▼
write one section per question from its notes ─► executive summary ─► citation checks ─► report
```

### How citations are guaranteed

1. **Grounded notes.** Each note the model extracts must include a verbatim quote from the page.
   The quote is checked against the fetched text (exact or near-exact match); notes that fail are
   discarded, so invented facts don't enter the note set.
2. **Tagged writing.** Sections are written only from notes, each tagged with its source id
   (`[S12]`), and the model is told to tag every factual sentence.
3. **Deterministic post-processing** (`deep_research/report.py`):
   - citations to source ids that don't exist or weren't used are removed;
   - paragraphs, list items and trailing sentences with no citation are **dropped**
     (or marked *[citation needed]* with `UNCITED_POLICY=flag`), and the removal is logged in the activity feed;
   - citations are renumbered `[1]`, `[2]`… by first appearance;
   - the **References** list (title, site, URL, access date) is generated from the database, never by the model.

The final report also states which model produced it and lists any research questions that
could not be answered.

A local model can still misread a source, so a cited sentence is not a guaranteed-correct one.
The citations exist so you can check claims quickly; click a citation number to jump to its reference.

## Configuration

All settings are environment variables (or `.env`); see [`.env.example`](.env.example). The ones you're most likely to change:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen3:8b` | Default model (selectable per topic) |
| `LLM_CONTEXT_CHARS` | `12000` | Text budget per model call |
| `SEARCH_PROVIDER` | `duckduckgo` | `searxng`, `duckduckgo` or `brave` |
| `MAX_ITERATIONS` | `3` | Research rounds (plan + follow-ups) |
| `MAX_QUESTIONS` | `5` | Initial questions / follow-ups per round |
| `MAX_SOURCES` | `25` | Total sources that contribute notes |
| `SOURCES_PER_QUESTION` | `3` | Sources read per question per round |
| `UNCITED_POLICY` | `drop` | `drop` or `flag` uncited passages |
| `DATA_DIR` | `data` | Where the SQLite database lives |

Run on another interface/port with `python -m deep_research --host 0.0.0.0 --port 8080`.
There is no authentication, so only expose it on a trusted network.

## API

The UI is a thin client over a JSON API:

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Model server status, available models, defaults |
| `GET /api/topics` | List topics |
| `POST /api/topics` | Create & queue `{title, description?, model?, max_iterations?, max_questions?, max_sources?, sources_per_question?}` |
| `GET /api/topics/{id}` | Topic with questions, sources, report (Markdown + rendered HTML) |
| `GET /api/topics/{id}/events?after=N` | Activity log (incremental) |
| `GET /api/topics/{id}/notes` | Extracted notes with their quotes and sources |
| `GET /api/topics/{id}/report.md` | Download the report |
| `POST /api/topics/{id}/cancel` · `/retry` | Cancel a run · re-run from scratch |
| `DELETE /api/topics/{id}` | Delete |

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use a fake model, search engine and web (`tests/fakes.py`), so they run offline and
exercise the full pipeline, including citation enforcement and quote verification.

Project layout:

```
deep_research/
  app.py      FastAPI routes + static UI
  worker.py   background queue (one research run at a time by default)
  agent.py    the research loop and report writer
  report.py   citation validation, renumbering, references, HTML rendering
  search.py   search providers, page fetching and text extraction
  llm.py      OpenAI-compatible client, JSON parsing, <think> stripping
  db.py       SQLite storage
  static/     index.html, app.js, styles.css
```

Limitations: PDFs and JavaScript-only pages are skipped; a run interrupted by a server restart is
marked failed and can be retried from scratch.
