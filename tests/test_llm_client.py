"""LLMClient against a faked HTTP model server (httpx.MockTransport)."""

import json

import httpx
import pytest

from deep_research.config import Settings
from deep_research.llm import LLMClient, LLMError


def reply(content="ok", *, finish="stop", usage=None, **message_extra):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content, **message_extra}, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
    })


class Server:
    """Records requests and plays back a scripted list of responses (or exceptions)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx.Request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.url.path, body))
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        return r


def client(server, settings=None, **kw):
    settings = settings or Settings()
    sleeps = []
    c = LLMClient("http://llm.test/v1", "m", profiles=settings.call_profiles(),
                  think_control=kw.pop("think_control", settings.llm_think_control),
                  transport=httpx.MockTransport(server), sleep=sleeps.append, **kw)
    c.sleeps = sleeps
    return c


def user(text="hi"):
    return [{"role": "user", "content": text}]


def test_fast_mode_turns_thinking_off_and_caps_output():
    srv = Server(reply())
    client(srv).chat(user(), mode="fast")
    body = srv.requests[0][1]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in body
    assert body["max_tokens"] == 4096 and body["temperature"] == 0.3


def test_deep_mode_uses_effort_level_and_qwen_sampling():
    srv = Server(reply())
    client(srv, Settings(llm_preset="qwen", llm_think_deep="medium")).chat(user(), mode="deep")
    body = srv.requests[0][1]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["reasoning_effort"] == "medium"
    assert body["temperature"] == 1.0 and body["top_p"] == 0.95 and body["top_k"] == 20
    assert body["max_tokens"] == 8192


def test_qwen_fast_sampling_and_overrides():
    s = Settings(llm_preset="qwen", llm_sampling_fast='{"temperature": 0.5}')
    srv = Server(reply())
    client(srv, s).chat(user(), mode="fast")
    body = srv.requests[0][1]
    assert body["temperature"] == 0.5 and body["presence_penalty"] == 1.5 and body["top_p"] == 0.8


@pytest.mark.parametrize("control,expect_ctk,expect_effort", [
    ("chat_template_kwargs", True, None),
    ("reasoning_effort", False, "none"),
    ("none", False, None),
])
def test_think_control_variants(control, expect_ctk, expect_effort):
    srv = Server(reply())
    client(srv, think_control=control).chat(user(), mode="fast")
    body = srv.requests[0][1]
    assert ("chat_template_kwargs" in body) == expect_ctk
    assert body.get("reasoning_effort") == expect_effort


def test_default_thinking_sends_no_thinking_fields():
    srv = Server(reply())
    client(srv, Settings(llm_think_deep="default")).chat(user(), mode="deep")
    body = srv.requests[0][1]
    assert "chat_template_kwargs" not in body and "reasoning_effort" not in body


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        Settings(llm_preset="nope").call_profiles()
    with pytest.raises(ValueError):
        Settings(llm_think_fast="maybe").call_profiles()


def test_400_falls_back_to_minimal_request_and_remembers():
    srv = Server(httpx.Response(400, text="unknown field chat_template_kwargs"), reply("a"), reply("b"))
    c = client(srv)
    assert c.chat(user(), json_mode=True) == "a"
    assert c.chat(user(), json_mode=True) == "b"
    first, retry, later = (b for _, b in srv.requests)
    assert "chat_template_kwargs" in first and "response_format" in first
    for body in (retry, later):
        assert not {"chat_template_kwargs", "reasoning_effort", "response_format", "top_k"} & body.keys()
        assert body["max_tokens"] == 4096
    assert any("rejected optional request fields" in w for w in c.pop_warnings())


def test_400_on_minimal_request_is_an_error():
    srv = Server(httpx.Response(400, text="bad"))
    c = client(srv)
    with pytest.raises(LLMError, match="400"):
        c.chat(user())
    assert len(srv.requests) == 2  # original + one minimal retry


def test_transient_errors_are_retried_with_backoff():
    srv = Server(httpx.ConnectError("refused"), httpx.Response(503, text="loading"), reply("fine"))
    c = client(srv)
    assert c.chat(user()) == "fine"
    assert len(srv.requests) == 3 and c.sleeps == [2.0, 4.0]


def test_retries_exhausted():
    srv = Server(httpx.ConnectError("refused"))
    c = client(srv)
    with pytest.raises(LLMError, match="Could not reach"):
        c.chat(user())
    assert len(srv.requests) == 3

    srv = Server(httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="500"):
        client(srv).chat(user())


def test_timeouts_are_not_retried():
    srv = Server(httpx.ReadTimeout("slow"))
    with pytest.raises(LLMError, match="timed out"):
        client(srv).chat(user())
    assert len(srv.requests) == 1


def test_output_budget_spent_thinking_is_reported():
    srv = Server(reply("<think>" + "hmm " * 50, finish="length"))
    with pytest.raises(LLMError, match="LLM_MAX_TOKENS_DEEP"):
        client(srv).chat(user(), mode="deep")


def test_empty_and_malformed_responses():
    with pytest.raises(LLMError, match="empty"):
        client(Server(reply(""))).chat(user())
    with pytest.raises(LLMError, match="Unexpected response"):
        client(Server(httpx.Response(200, json={"nope": 1}))).chat(user())


def test_reasoning_is_stripped_and_flagged_when_thinking_should_be_off():
    srv = Server(reply("<think>let me think</think>answer"), reply("again", reasoning_content="more thoughts"))
    c = client(srv)
    assert c.chat(user(), mode="fast") == "answer"
    assert c.chat(user(), mode="fast") == "again"
    warnings = c.pop_warnings()
    assert len(warnings) == 1 and "still thinking" in warnings[0]  # reported once


def test_thinking_in_deep_mode_is_not_flagged():
    c = client(Server(reply("<think>fine</think>answer")))
    c.chat(user(), mode="deep")
    assert c.pop_warnings() == []


def test_truncated_prompt_is_detected():
    srv = Server(reply("x", usage={"prompt_tokens": 300, "completion_tokens": 1}))
    c = client(srv)
    c.chat(user("word " * 4000))  # ~20k chars, server saw only 300 tokens
    assert any("cut the prompt short" in w for w in c.pop_warnings())


def test_usage_stats_and_with_model_isolation():
    srv = Server(reply(usage={"prompt_tokens": 100, "completion_tokens": 20}))
    base = client(srv)
    run = base.with_model("other")
    run.chat(user())
    run.chat(user())
    assert run.stats["calls"] == 2 and run.stats["prompt_tokens"] == 200 and run.stats["completion_tokens"] == 40
    assert base.stats["calls"] == 0
    assert srv.requests[0][1]["model"] == "other"


def test_chat_json_retries_invalid_json():
    srv = Server(reply("not json"), reply('{"a": 1}'))
    c = client(srv)
    assert c.chat_json("sys", "give json") == {"a": 1}
    retry_msgs = srv.requests[1][1]["messages"]
    assert retry_msgs[-1]["content"].startswith("That was not valid JSON")
    assert srv.requests[0][1]["response_format"] == {"type": "json_object"}


def test_chat_json_gives_up():
    with pytest.raises(LLMError, match="valid JSON"):
        client(Server(reply("still not json"))).chat_json("sys", "x", retries=1)


def test_list_models():
    srv = Server(httpx.Response(200, json={"data": [{"id": "b"}, {"id": "a"}]}))
    assert client(srv).list_models() == ["a", "b"]
    assert srv.requests[0][0] == "/v1/models"
