my"""Tests for LLM / OpenAI-compatible strategy advisor helpers."""

from __future__ import annotations

from daytrading.analytics.llm_strategy_advisor import (
    LLMStrategyAdvisor,
    endpoint_requires_api_key,
    parse_adaptation_json,
)


def test_endpoint_requires_api_key() -> None:
    assert endpoint_requires_api_key("https://api.openai.com/v1")
    assert endpoint_requires_api_key("https://my.openai.azure.com/openai/v1")
    assert not endpoint_requires_api_key("http://127.0.0.1:11434/v1")
    assert not endpoint_requires_api_key("http://localhost:11434/v1")


def test_parse_adaptation_json_fence() -> None:
    text = '```json\n{"rationale": "x", "insights": []}\n```'
    d = parse_adaptation_json(text)
    assert d["rationale"] == "x"


def test_json_object_mode_local_default() -> None:
    adv = LLMStrategyAdvisor(
        api_key="",
        model="llama3.2",
        base_url="http://127.0.0.1:11434/v1",
        json_object_mode=None,
    )
    assert adv._use_json_object_response_format() is False


def test_json_object_mode_openai_default() -> None:
    adv = LLMStrategyAdvisor(
        api_key="sk-test",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        json_object_mode=None,
    )
    assert adv._use_json_object_response_format() is True


def test_json_object_mode_override() -> None:
    adv = LLMStrategyAdvisor(
        api_key="",
        model="x",
        base_url="http://127.0.0.1:11434/v1",
        json_object_mode=True,
    )
    assert adv._use_json_object_response_format() is True
