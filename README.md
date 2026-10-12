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
OLLAMA_CONTEXT_LENGTH=16384 ollama serve     # Ollama's default context is too small; see below

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
Small) work; bigger ones write noticeably better reports.

**Context window.** `LLM_CONTEXT_CHARS` (default 24000) caps how much text goes into one call.
The server's context window must hold that (about 3.5 characters per token) *plus* the reply,
including any thinking. If it doesn't, Ollama silently drops the start of the prompt, which
includes the instructions. The app warns in the activity log when it detects this. Rule of
thumb: context tokens ≥ `LLM_CONTEXT_CHARS / 3.5 + LLM_MAX_TOKENS_DEEP`.

**Thinking models.** Every model call is either *fast* (writing search queries, ranking results,
extracting notes: mechanical, run with thinking off) or *deep* (planning, reviewing coverage,
writing: thinking at `low` by default). Long thinking on every call is the main reason a run
takes hours instead of minutes. Thinking is requested two ways by default
(`LLM_THINK_CONTROL=both`): `chat_template_kwargs.enable_thinking`, honoured by vLLM, SGLang
and llama.cpp's `llama-server` (started with `--jinja`), and `reasoning_effort`. If the activity
log says *"The model is still thinking although thinking was turned off"*, your server ignores
both. Try `LLM_THINK_CONTROL=reasoning_effort`, or use one of the servers above. If a server
rejects these fields, the app drops them and carries on (and logs that it did).

### Recommended setup for Qwen 3.8 27B

Qwen 3.8 27B (dense, 262K context, ~17 GB at Q4_K_M) is a strong choice, but it thinks at
maximum effort by default, which is very slow at the 15–30 tokens/s typical of local hardware.
Use the Qwen preset, which applies the model card's sampling settings, and give it a real context
window. For example, with llama.cpp:

```bash
llama-server -m qwen3.8-27b-q4_k_m.gguf --jinja --ctx-size 32768 --port 8080
```

```ini
# .env
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=qwen3.8-27b          # whatever name your server reports (shown in the UI)
LLM_PRESET=qwen
LLM_CONTEXT_CHARS=60000        # whole pages fit in one call (MAX_PAGE_CHARS is 40000)
LLM_THINK_FAST=off
LLM_THINK_DEEP=low             # try medium for better plans/writing if you can wait
```

32K tokens of context covers 60000 characters (~17K tokens) plus the 8192-token deep output cap.
With Ollama, set `OLLAMA_CONTEXT_LENGTH=32768` instead. The *Model usage* line at the end of
each run's activity log shows calls, tokens and model time, so you can tune from real numbers.

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
outline: model groups all notes into 3–7 themed sections (merging overlapping questions)
  ─► write each section from its notes, knowing the whole outline
  ─► citation checks + verification ─► executive summary (sees every section) ─► report
```

The outline step is what turns per-question research into a readable report: background first,
then findings, comparisons and open issues, with overlapping material merged. The code validates
the outline (only real notes, each used once). If the model can't produce a usable outline, the
report falls back to one section per research question.

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
4. **Citation verification.** Each section and the summary are split into *cited statements*: a
   citation plus all the text since the previous one, so uncited sentences before a citation are
   checked too. The model checks every statement against the notes and verbatim quotes from the
   sources it cites:
   - **supported**: kept;
   - **wrong source**: the citation is replaced with the source that does support it (only a
     source present in the evidence is accepted);
   - **unsupported** (not backed up, or adds numbers, dates, causes or certainty the evidence lacks):
     removed, or marked *[unverified]* with `UNCITED_POLICY=flag`.

   The activity log lists each removed statement and the reason. The report's *Scope and
   limitations* section states how many statements were checked, confirmed, re-attributed and
   removed. If a check can't run, the text is kept and counted as unchecked. This costs one
   extra model call per section; turn it off with `VERIFY_CITATIONS=false`.
5. **Safe rendering.** Model-written text is untrusted, since web pages can try to inject instructions.
   Raw HTML is escaped, images are removed, and only `http(s)` links survive in the rendered report.

The final report also states which model produced it and lists any research questions that
could not be answered.

Verification is done by the same local model, so it reduces errors but can't eliminate them: a
model can wrongly confirm a statement or wrongly reject a good one. The citations exist so you can
check claims quickly; click a citation number to jump to its reference.

## Configuration

All settings are environment variables (or `.env`); see [`.env.example`](.env.example). The ones you're most likely to change:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen3:8b` | Default model (selectable per topic) |
| `LLM_CONTEXT_CHARS` | `24000` | Text budget per model call (see *Context window*) |
| `LLM_PRESET` | `generic` | Sampling settings: `generic` or `qwen` (Qwen model-card values) |
| `LLM_THINK_FAST` / `LLM_THINK_DEEP` | `off` / `low` | Thinking per call type: `off`, `low`, `medium`, `high`, `xhigh`, `default` (send nothing) |
| `LLM_THINK_CONTROL` | `both` | How thinking is requested: `both`, `chat_template_kwargs`, `reasoning_effort`, `none` |
| `LLM_MAX_TOKENS_FAST` / `_DEEP` | `4096` / `8192` | Output cap per call, thinking included |
| `LLM_SAMPLING_FAST` / `_DEEP` | – | JSON overrides, e.g. `{"temperature": 0.5}` |
| `LLM_TIMEOUT` | `900` | Seconds per model call |
| `LLM_MAX_CONSECUTIVE_FAILURES` | `3` | Stop a run after this many failed calls in a row |
| `SEARCH_PROVIDER` | `duckduckgo` | `searxng`, `duckduckgo` or `brave` |
| `MAX_ITERATIONS` | `3` | Research rounds (plan + follow-ups) |
| `MAX_QUESTIONS` | `5` | Initial questions / follow-ups per round |
| `MAX_SOURCES` | `25` | Total sources that contribute notes |
| `SOURCES_PER_QUESTION` | `3` | Sources read per question per round |
| `VERIFY_CITATIONS` | `true` | Check each cited statement against its sources' notes |
| `UNCITED_POLICY` | `drop` | `drop` or `flag` passages that are uncited or fail verification |
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

Tests run offline. The pipeline tests use a fake model, search engine and web
(`tests/fakes.py`). The model client, search providers and fetcher are tested against faked HTTP
responses (`httpx.MockTransport`), covering retries, fallbacks and error handling. Check coverage
with `pip install pytest-cov && pytest --cov=deep_research --cov-branch`.

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

A failed model call (timeout, server error, unusable output) costs only that step. The page,
section or review is skipped with a warning in the activity log, and the run continues. A run
stops only when `LLM_MAX_CONSECUTIVE_FAILURES` calls fail in a row, which usually means the model
server is down.

Limitations: PDFs and JavaScript-only pages are skipped; a run interrupted by a server restart is
marked failed and can be retried from scratch.
