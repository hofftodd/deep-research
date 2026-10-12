import pytest

from deep_research.llm import LLMError, parse_json, strip_reasoning


def test_strip_reasoning():
    assert strip_reasoning("<think>hmm</think>\nAnswer") == "Answer"
    assert strip_reasoning("leftover thought</think>Answer") == "Answer"


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure!\n```json\n{"a": [1, 2,]}\n```') == {"a": [1, 2]}
    assert parse_json('<think>{"no": 1}</think>Here: {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json("no json here")
