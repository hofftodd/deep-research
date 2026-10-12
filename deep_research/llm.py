"""Client for a local model served through an OpenAI-compatible chat completions API.

Works with Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1),
llama.cpp's llama-server, vLLM, LocalAI, etc.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    pass


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (qwen3, deepseek-r1...)."""
    text = _THINK_RE.sub("", text)
    # An unterminated think block (output truncated mid-thought) leaves nothing usable before it.
    if "</think>" in text.lower():
        text = text[text.lower().rindex("</think>") + len("</think>"):]
    return text.strip()


def parse_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from model output."""
    text = strip_reasoning(text)
    candidates = [text]
    candidates += [m.strip() for m in _FENCE_RE.findall(text)]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            # Trailing commas are a common small-model mistake.
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", c))
            except (json.JSONDecodeError, ValueError):
                continue
    raise LLMError(f"Model did not return valid JSON: {text[:300]!r}")


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "local",
                 timeout: float = 600.0, temperature: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def with_model(self, model: str) -> "LLMClient":
        clone = object.__new__(LLMClient)
        clone.__dict__.update(self.__dict__)
        clone.model = model
        return clone

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False,
             max_tokens: int | None = None, temperature: float | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = self._http.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach model server at {self.base_url}: {exc}") from exc
        if resp.status_code == 400 and json_mode:
            # Some servers don't support response_format; retry without it.
            return self.chat(messages, json_mode=False, max_tokens=max_tokens, temperature=temperature)
        if resp.status_code >= 400:
            raise LLMError(f"Model server returned {resp.status_code}: {resp.text[:300]}")
        try:
            content = resp.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Unexpected response from model server: {resp.text[:300]}") from exc
        return strip_reasoning(content)

    def chat_json(self, system: str, user: str, retries: int = 2) -> Any:
        messages = [
            {"role": "system", "content": system + "\n\nRespond with a single valid JSON object and nothing else."},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for _ in range(retries + 1):
            raw = self.chat(messages, json_mode=True)
            try:
                return parse_json(raw)
            except LLMError as exc:
                last_err = exc
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw[:2000]},
                    {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object."},
                ]
        raise last_err  # type: ignore[misc]

    def list_models(self) -> list[str]:
        resp = self._http.get(f"{self.base_url}/models", timeout=5.0)
        resp.raise_for_status()
        return sorted(m["id"] for m in resp.json().get("data", []))
