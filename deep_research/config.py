"""Runtime configuration, read from environment variables (optionally a .env file)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

THINKING_LEVELS = ("off", "low", "medium", "high", "xhigh", "default")

# Sampling presets per call mode. "fast" calls are mechanical (search queries, ranking results,
# extracting notes) and run without thinking; "deep" calls (planning, coverage review, writing)
# may think. The qwen preset follows the Qwen 3.x model cards: non-thinking temperature 0.7 /
# top_p 0.8 / presence_penalty 1.5, thinking temperature 1.0 / top_p 0.95.
SAMPLING_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "generic": {
        "fast": {"temperature": 0.3, "top_p": 0.9},
        "deep": {"temperature": 0.6, "top_p": 0.95},
    },
    "qwen": {
        "fast": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
        "deep": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
    },
}


@dataclass
class CallProfile:
    """How the model is called for one mode ("fast" or "deep")."""
    thinking: str
    max_tokens: int
    sampling: dict[str, Any]


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
    llm_timeout: float = 900.0
    # Approximate context budget (in characters) for text sent to the model in one call.
    # The model server's context window must hold this plus the reply (about 3.5 chars/token).
    llm_context_chars: int = 24000
    # Sampling preset ("generic" or "qwen") plus optional JSON overrides per mode.
    llm_preset: str = "generic"
    llm_sampling_fast: str = ""
    llm_sampling_deep: str = ""
    # Thinking level per mode: off | low | medium | high | xhigh | default (send nothing).
    llm_think_fast: str = "off"
    llm_think_deep: str = "low"
    # How thinking is requested: "both", "chat_template_kwargs", "reasoning_effort" or "none".
    llm_think_control: str = "both"
    # Output cap per call (thinking tokens count towards it on most servers).
    llm_max_tokens_fast: int = 4096
    llm_max_tokens_deep: int = 8192
    # Give up on a run after this many model calls fail in a row (server down, etc.).
    llm_max_consecutive_failures: int = 3

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

    def call_profiles(self) -> dict[str, CallProfile]:
        if self.llm_preset not in SAMPLING_PRESETS:
            raise ValueError(f"LLM_PRESET must be one of {sorted(SAMPLING_PRESETS)}, got {self.llm_preset!r}")
        for name, level in (("LLM_THINK_FAST", self.llm_think_fast), ("LLM_THINK_DEEP", self.llm_think_deep)):
            if level not in THINKING_LEVELS:
                raise ValueError(f"{name} must be one of {THINKING_LEVELS}, got {level!r}")
        preset = SAMPLING_PRESETS[self.llm_preset]
        out = {}
        for mode, think, cap, override in (
            ("fast", self.llm_think_fast, self.llm_max_tokens_fast, self.llm_sampling_fast),
            ("deep", self.llm_think_deep, self.llm_max_tokens_deep, self.llm_sampling_deep),
        ):
            sampling = dict(preset[mode])
            if override.strip():
                sampling.update(json.loads(override))
            out[mode] = CallProfile(thinking=think, max_tokens=cap, sampling=sampling)
        return out

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(Path(os.environ.get("DR_ENV_FILE", ".env")))
        e = os.environ.get
        return cls(
            llm_base_url=e("LLM_BASE_URL", cls.llm_base_url).rstrip("/"),
            llm_model=e("LLM_MODEL", cls.llm_model),
            llm_api_key=e("LLM_API_KEY", cls.llm_api_key),
            llm_timeout=_float("LLM_TIMEOUT", cls.llm_timeout),
            llm_context_chars=_int("LLM_CONTEXT_CHARS", cls.llm_context_chars),
            llm_preset=e("LLM_PRESET", cls.llm_preset).lower(),
            llm_sampling_fast=e("LLM_SAMPLING_FAST", cls.llm_sampling_fast),
            llm_sampling_deep=e("LLM_SAMPLING_DEEP", cls.llm_sampling_deep),
            llm_think_fast=e("LLM_THINK_FAST", cls.llm_think_fast).lower(),
            llm_think_deep=e("LLM_THINK_DEEP", cls.llm_think_deep).lower(),
            llm_think_control=e("LLM_THINK_CONTROL", cls.llm_think_control).lower(),
            llm_max_tokens_fast=_int("LLM_MAX_TOKENS_FAST", cls.llm_max_tokens_fast),
            llm_max_tokens_deep=_int("LLM_MAX_TOKENS_DEEP", cls.llm_max_tokens_deep),
            llm_max_consecutive_failures=_int("LLM_MAX_CONSECUTIVE_FAILURES", cls.llm_max_consecutive_failures),
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
