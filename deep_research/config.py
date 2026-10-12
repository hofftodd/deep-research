"""Runtime configuration, read from environment variables (optionally a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    # Local LLM: any OpenAI-compatible server (Ollama, LM Studio, llama.cpp server, vLLM...).
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:8b"
    llm_api_key: str = "local"
    llm_timeout: float = 600.0
    llm_temperature: float = 0.2
    # Approximate context budget (in characters) for text sent to the model in one call.
    llm_context_chars: int = 12000

    # Web search provider: "searxng", "duckduckgo" or "brave".
    search_provider: str = "duckduckgo"
    searxng_url: str = "http://localhost:8888"
    brave_api_key: str = ""
    search_results_per_query: int = 8

    # Research defaults (can be overridden per topic).
    max_iterations: int = 3
    max_questions: int = 5
    max_sources: int = 25
    sources_per_question: int = 3
    queries_per_question: int = 2

    # Fetching
    fetch_timeout: float = 20.0
    max_page_chars: int = 40000
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 DeepResearch/1.0"
    )

    # What to do with report paragraphs that carry no citation: "drop" or "flag".
    uncited_policy: str = "drop"

    data_dir: Path = field(default_factory=lambda: Path("data"))
    worker_concurrency: int = 1

    @property
    def db_path(self) -> Path:
        return self.data_dir / "research.db"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(Path(os.environ.get("DR_ENV_FILE", ".env")))
        e = os.environ.get
        return cls(
            llm_base_url=e("LLM_BASE_URL", cls.llm_base_url).rstrip("/"),
            llm_model=e("LLM_MODEL", cls.llm_model),
            llm_api_key=e("LLM_API_KEY", cls.llm_api_key),
            llm_timeout=_float("LLM_TIMEOUT", cls.llm_timeout),
            llm_temperature=_float("LLM_TEMPERATURE", cls.llm_temperature),
            llm_context_chars=_int("LLM_CONTEXT_CHARS", cls.llm_context_chars),
            search_provider=e("SEARCH_PROVIDER", cls.search_provider).lower(),
            searxng_url=e("SEARXNG_URL", cls.searxng_url).rstrip("/"),
            brave_api_key=e("BRAVE_API_KEY", cls.brave_api_key),
            search_results_per_query=_int("SEARCH_RESULTS_PER_QUERY", cls.search_results_per_query),
            max_iterations=_int("MAX_ITERATIONS", cls.max_iterations),
            max_questions=_int("MAX_QUESTIONS", cls.max_questions),
            max_sources=_int("MAX_SOURCES", cls.max_sources),
            sources_per_question=_int("SOURCES_PER_QUESTION", cls.sources_per_question),
            queries_per_question=_int("QUERIES_PER_QUESTION", cls.queries_per_question),
            fetch_timeout=_float("FETCH_TIMEOUT", cls.fetch_timeout),
            max_page_chars=_int("MAX_PAGE_CHARS", cls.max_page_chars),
            uncited_policy=e("UNCITED_POLICY", cls.uncited_policy).lower(),
            data_dir=Path(e("DATA_DIR", "data")),
            worker_concurrency=_int("WORKER_CONCURRENCY", cls.worker_concurrency),
        )
