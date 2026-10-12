"""Client for a local model served through an OpenAI-compatible chat completions API.

Works with Ollama (http://localhost:11434/v1), LM Studio (http://localhost:1234/v1),
llama.cpp's llama-server, vLLM, LocalAI, etc.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

import httpx

from .config import CallProfile, Settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    pass


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (qwen3, deepseek-r1...)."""
    text = _THINK_RE.sub("", text)
    # A stray </think> (the opening tag was part of the chat template) ends the reasoning.
    if "</think>" in text.lower():
        text = text[text.lower().rindex("</think>") + len("</think>"):]
    # An unclosed <think> means the output was cut off mid-thought: nothing after it is an answer.
    if "<think>" in text.lower():
        text = text[: text.lower().index("<think>")]
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


# Request fields beyond the core OpenAI chat API. Servers that reject them (HTTP 400) get a
# minimal request instead, and the client remembers not to send them again.
OPTIONAL_FIELDS = ("response_format", "chat_template_kwargs", "reasoning_effort", "top_k",
                   "presence_penalty", "min_p")


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: float = 900.0,
                 profiles: dict[str, CallProfile] | None = None, think_control: str = "both",
                 transport: httpx.BaseTransport | None = None, retries: int = 2,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.profiles = profiles or Settings().call_profiles()
        self.think_control = think_control
        self.retries = retries
        self._sleep = sleep
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
        )
        # Shared between clones so a server's quirks are learned once.
        self._compat: dict[str, bool] = {"minimal": False}
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        self.warnings: list[str] = []
        self._warned: set[str] = set()

    def with_model(self, model: str) -> "LLMClient":
        """A client for one research run: same connection and settings, its own usage stats."""
        clone = object.__new__(LLMClient)
        clone.__dict__.update(self.__dict__)
        clone.model = model
        clone._reset_tracking()
        return clone

    def pop_warnings(self) -> list[str]:
        out, self.warnings = self.warnings, []
        return out

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.warnings.append(message)

    def _payload(self, messages: list[dict[str, str]], mode: str, json_mode: bool) -> dict[str, Any]:
        prof = self.profiles[mode]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": prof.max_tokens,
            **prof.sampling,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if prof.thinking != "default" and self.think_control != "none":
            if self.think_control in ("both", "chat_template_kwargs"):
                # vLLM, SGLang, llama.cpp's llama-server
                payload["chat_template_kwargs"] = {"enable_thinking": prof.thinking != "off"}
            if self.think_control in ("both", "reasoning_effort"):
                if prof.thinking != "off":
                    payload["reasoning_effort"] = prof.thinking
                elif self.think_control == "reasoning_effort":
                    payload["reasoning_effort"] = "none"
        if self._compat["minimal"]:
            for key in OPTIONAL_FIELDS:
                payload.pop(key, None)
        return payload

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """POST with retries for transient failures. Timeouts are not retried: a call that ran
        for LLM_TIMEOUT seconds would most likely do so again."""
        attempt = 0
        while True:
            try:
                resp = self._http.post(f"{self.base_url}/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                raise LLMError(f"Model call timed out ({type(exc).__name__}); consider raising LLM_TIMEOUT "
                               "or lowering thinking/max tokens") from exc
            except httpx.TransportError as exc:
                if attempt >= self.retries:
                    raise LLMError(f"Could not reach model server at {self.base_url}: {exc}") from exc
            else:
                if resp.status_code < 500 or attempt >= self.retries:
                    return resp
            attempt += 1
            self._sleep(2.0 * attempt)

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False, mode: str = "fast") -> str:
        payload = self._payload(messages, mode, json_mode)
        started = time.monotonic()
        resp = self._post(payload)
        if resp.status_code == 400 and not self._compat["minimal"] and any(k in payload for k in OPTIONAL_FIELDS):
            self._compat["minimal"] = True
            self._warn_once("minimal", "Model server rejected optional request fields (JSON mode, thinking "
                            f"control, extra sampling); continuing without them. Server said: {resp.text[:200]}")
            payload = self._payload(messages, mode, json_mode)
            resp = self._post(payload)
        if resp.status_code >= 400:
            raise LLMError(f"Model server returned {resp.status_code}: {resp.text[:300]}")
        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"Unexpected response from model server: {resp.text[:300]}") from exc

        content = message.get("content") or ""
        usage = data.get("usage") or {}
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += usage.get("prompt_tokens") or 0
        self.stats["completion_tokens"] += usage.get("completion_tokens") or 0
        self.stats["seconds"] += time.monotonic() - started
        self._check_diagnostics(messages, mode, message, content, usage)

        text = strip_reasoning(content)
        if not text:
            if choice.get("finish_reason") == "length":
                raise LLMError(f"Model used its whole output budget ({payload.get('max_tokens')} tokens) without "
                               f"answering, probably while thinking; raise LLM_MAX_TOKENS_{mode.upper()} or "
                               f"lower LLM_THINK_{mode.upper()}")
            raise LLMError("Model returned an empty response")
        return text

    def _check_diagnostics(self, messages, mode, message, content, usage) -> None:
        thought = bool(message.get("reasoning_content") or message.get("reasoning")) or "</think>" in content.lower()
        if thought and self.profiles[mode].thinking == "off":
            self._warn_once("think-off", "The model is still thinking although thinking was turned off for fast "
                            "steps; your server may not support the requested control. This makes research much "
                            "slower. See LLM_THINK_CONTROL in the README.")
        prompt_tokens = usage.get("prompt_tokens") or 0
        expected = sum(len(m.get("content") or "") for m in messages) / 4.5
        if prompt_tokens and expected > 1000 and prompt_tokens < expected * 0.5:
            self._warn_once("truncated", f"The model server seems to have cut the prompt short ({prompt_tokens} "
                            f"tokens received, ~{int(expected)}+ expected). Its context window is probably too "
                            "small: raise it (e.g. OLLAMA_CONTEXT_LENGTH) or lower LLM_CONTEXT_CHARS.")

    def chat_json(self, system: str, user: str, mode: str = "fast", retries: int = 2) -> Any:
        messages = [
            {"role": "system", "content": system + "\n\nRespond with a single valid JSON object and nothing else."},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for _ in range(retries + 1):
            raw = self.chat(messages, json_mode=True, mode=mode)
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
