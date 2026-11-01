"""Prove token budgeting and cache reuse using mocked provider responses only."""

import json
import sqlite3
from unittest.mock import Mock

import pytest
from pydantic import Field

from app.config import Config
from app.knowledge import KnowledgeStore
from app.llm.adapter import LLMClient, LLMError
from app.llm.outputs import StructuredOutput
from app.llm.session import OptimizationSession


class Answer(StructuredOutput):
    value: str


class AnswerV2(StructuredOutput):
    value: str = Field(min_length=3)


def _response(value=None, usage=None, *, status=200):
    return Mock(status_code=status, headers={}, json=Mock(return_value={
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps(value or {"value": "expected"})}],
        "usage": usage or {},
    }))


def _client(**kwargs):
    return LLMClient(api_base="https://api.anthropic.com/v1", api_key="mock-only-key",
                     model=kwargs.pop("model", "claude-sonnet-5-5"), max_retries=0,
                     max_tokens=kwargs.pop("max_tokens", 8192), **kwargs)


def _request(session, **changes):
    arguments = dict(stage="plan", system="Use verified facts only.",
                     user='{"facts":["Built Python services"]}', output_model=Answer,
                     max_tokens=2048)
    arguments.update(changes)
    return session.request(**arguments)


@pytest.fixture
def store():
    database = KnowledgeStore(":memory:")
    yield database
    database.close()


@pytest.fixture(autouse=True)
def deterministic_settings(monkeypatch):
    monkeypatch.setattr(Config, "CLAUDE_EFFORT", "medium")
    monkeypatch.setattr(Config, "CLAUDE_PROMPT_CACHING", True)


def test_exact_validated_response_is_reused_across_runs_without_provider_call(store, monkeypatch):
    post = Mock(return_value=_response(usage={"input_tokens": 120, "output_tokens": 20}))
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    first = OptimizationSession(_client(), store, "candidate")
    assert _request(first) == {"value": "expected"}
    second = OptimizationSession(_client(), store, "candidate")
    assert _request(second) == {"value": "expected"}
    assert post.call_count == 1
    assert first.usage()["api_calls"] == 1
    assert second.usage() == {
        "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0, "api_calls": 0, "local_cache_hits": 1,
        "model": "claude-sonnet-5-5", "total_input_tokens": 0,
    }


@pytest.mark.parametrize("change", [
    "facts", "model", "effort", "schema", "stage", "profile", "system",
    "endpoint", "output_budget", "prompt_version",
])
def test_cache_is_invalidated_by_every_generation_input(store, monkeypatch, change):
    post = Mock(return_value=_response())
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    _request(OptimizationSession(_client(), store))
    client = _client()
    profile = "default"
    changed_request = {}
    if change == "facts":
        changed_request["user"] = '{"facts":["Built Python and SQL services"]}'
    elif change == "model":
        client.model = "different-model"
    elif change == "effort":
        monkeypatch.setattr(Config, "CLAUDE_EFFORT", "low")
    elif change == "schema":
        changed_request["output_model"] = AnswerV2
    elif change == "stage":
        changed_request["stage"] = "refine"
    elif change == "profile":
        profile = "other-candidate"
    elif change == "system":
        changed_request["system"] = "Use verified facts and preserve exact dates."
    elif change == "endpoint":
        client.api_base = "https://mock-provider.example/v1"
    elif change == "output_budget":
        changed_request["max_tokens"] = 2049
    elif change == "prompt_version":
        monkeypatch.setattr("app.llm.session.PROMPT_VERSION", "knowledge-plan-next")
    second = OptimizationSession(client, store, profile)
    assert _request(second, **changed_request) == {"value": "expected"}
    assert post.call_count == 2
    assert second.usage()["local_cache_hits"] == 0


def test_capped_equivalent_output_budgets_share_cache(store, monkeypatch):
    post = Mock(return_value=_response())
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    session = OptimizationSession(_client(max_tokens=1024), store)
    _request(session, max_tokens=2048)
    _request(session, max_tokens=4096)
    assert post.call_count == 1
    assert post.call_args.kwargs["json"]["max_tokens"] == 1024
    assert session.usage()["local_cache_hits"] == 1


def test_corrupt_cached_json_is_revalidated_then_replaced(store, monkeypatch):
    post = Mock(return_value=_response())
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    _request(OptimizationSession(_client(), store))
    with store._db:
        store._db.execute("UPDATE response_cache SET payload_json=?", ('{"value":123}',))
    second = OptimizationSession(_client(), store)
    assert _request(second) == {"value": "expected"}
    assert post.call_count == 2
    assert second.usage()["local_cache_hits"] == 0
    assert _request(OptimizationSession(_client(), store)) == {"value": "expected"}
    assert post.call_count == 2


def test_malformed_cached_json_cannot_break_a_run(store, monkeypatch):
    post = Mock(return_value=_response())
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    _request(OptimizationSession(_client(), store))
    with store._db:
        store._db.execute("UPDATE response_cache SET payload_json=?", ("not valid JSON",))
    assert _request(OptimizationSession(_client(), store)) == {"value": "expected"}
    assert post.call_count == 2


def test_invalid_provider_response_is_never_cached(store, monkeypatch):
    post = Mock(side_effect=[_response({"value": 123}), _response()])
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    session = OptimizationSession(_client(), store)
    with pytest.raises(LLMError, match="invalid structured"):
        _request(session)
    assert store._db.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0] == 0
    assert _request(session) == {"value": "expected"}
    assert post.call_count == 2


def test_cache_storage_failure_still_returns_validated_provider_output(monkeypatch):
    store = Mock()
    store.get_cached_response.side_effect = sqlite3.OperationalError("database unavailable")
    store.set_cached_response.side_effect = sqlite3.OperationalError("database unavailable")
    post = Mock(return_value=_response())
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    assert _request(OptimizationSession(_client(), store)) == {"value": "expected"}
    assert post.call_count == 1


def test_usage_sums_actual_provider_tokens_and_excludes_previous_runs(store, monkeypatch):
    post = Mock(side_effect=[
        _response(usage={"input_tokens": 100, "output_tokens": 10,
                         "cache_read_input_tokens": 200, "cache_creation_input_tokens": 300}),
        _response(usage={"input_tokens": 40, "output_tokens": 20,
                         "cache_read_input_tokens": 500, "cache_creation_input_tokens": 0}),
        _response(usage={"input_tokens": 60, "output_tokens": 30,
                         "cache_read_input_tokens": 100, "cache_creation_input_tokens": 50}),
    ])
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    client = _client()
    client.chat_json("previous-run", "previous", output_model=Answer)
    session = OptimizationSession(client, store)
    _request(session)
    _request(session, stage="rewrite")
    _request(session)  # Local reuse contributes no provider tokens.
    assert session.usage() == {
        "input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 600,
        "cache_creation_input_tokens": 50, "api_calls": 2, "local_cache_hits": 1,
        "model": "claude-sonnet-5-5", "total_input_tokens": 750,
    }
    assert post.call_count == 3


def test_usage_without_provider_counts_never_invents_token_savings(store, monkeypatch):
    post = Mock(return_value=_response())
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    session = OptimizationSession(_client(), store)
    _request(session)
    usage = session.usage()
    assert usage["api_calls"] == 1
    assert usage["total_input_tokens"] == 0
    assert usage["output_tokens"] == 0


def test_native_payload_uses_stable_prefix_cache_schema_and_stage_budget(monkeypatch):
    post = Mock(return_value=_response())
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    _request(OptimizationSession(_client(max_tokens=8192)), max_tokens=1536)
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "claude-sonnet-5-5"
    assert payload["max_tokens"] == 1536
    assert payload["system"] == [{"type": "text", "text": "Use verified facts only.",
                                  "cache_control": {"type": "ephemeral"}}]
    assert payload["output_config"]["effort"] == "medium"
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["messages"][0]["role"] == "user"
    assert "temperature" not in payload
