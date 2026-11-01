"""Claude wire protocol, strict output contracts, and pipeline integration."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from app import create_app
from app.config import Config
from app.llm.adapter import LLMClient, LLMError, get_llm_client
from app.llm.outputs import EvidenceOutput, JobOutput, RewriteOutput
from app.models.schemas import CompilationResult
from app.pipeline.orchestrator import run_optimization
from app.pipeline.orchestrator import _final_gaps
from app.pipeline.parse_resume import parse_resume_latex


def response(body=None, status=200):
    return Mock(status_code=status, headers={}, json=Mock(return_value=body))


def message(data, stop="end_turn"):
    return response({"stop_reason": stop, "content": [
        {"type": "text", "text": json.dumps(data)},
    ]})


def test_native_messages_payload_and_validation(monkeypatch):
    post = Mock(return_value=message({"mappings": []}))
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    client = LLMClient(api_key="private-test-key", model="test-claude", max_retries=0)
    assert client.chat_json("system rules", "source facts", output_model=EvidenceOutput) == {"mappings": []}
    args, kwargs = post.call_args
    assert args[0] == "https://api.anthropic.com/v1/messages"
    assert kwargs["headers"]["x-api-key"] == "private-test-key"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert kwargs["json"]["system"] == [{"type": "text", "text": "system rules", "cache_control": {"type": "ephemeral"}}]
    assert kwargs["json"]["max_tokens"] > 0
    assert kwargs["json"]["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["json"]["output_config"]["effort"] == "medium"
    assert "response_format" not in kwargs["json"]
    assert kwargs["allow_redirects"] is False


@pytest.mark.parametrize("status, expected", [(401, "authentication"), (403, "access"), (404, "not found")])
def test_permanent_errors_are_not_retried_or_leaked(monkeypatch, status, expected):
    post = Mock(return_value=response({"error": "private resume text"}, status))
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    with pytest.raises(LLMError, match=expected) as exc:
        LLMClient(api_key="secret-key").chat_json("system", "user")
    assert "private" not in str(exc.value) and "secret-key" not in str(exc.value)
    assert post.call_count == 1


def test_rate_limit_retry(monkeypatch):
    post = Mock(side_effect=[response(status=429), message({"mappings": []})])
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    monkeypatch.setattr("app.llm.adapter.time.sleep", Mock())
    assert LLMClient(api_key="test", max_retries=1).chat_json("s", "u") == {"mappings": []}
    assert post.call_count == 2


@pytest.mark.parametrize("stop", ["max_tokens", "refusal", "pause_turn", None])
def test_reject_incomplete_response(monkeypatch, stop):
    monkeypatch.setattr("app.llm.adapter.requests.post", Mock(return_value=message({"mappings": []}, stop)))
    with pytest.raises(LLMError):
        LLMClient(api_key="test").chat_json("s", "u", output_model=EvidenceOutput)


def test_invalid_schema_is_safe(monkeypatch):
    monkeypatch.setattr("app.llm.adapter.requests.post", Mock(return_value=message({"mappings": "private text"})))
    with pytest.raises(LLMError, match="invalid structured") as exc:
        LLMClient(api_key="test").chat_json("s", "u", output_model=EvidenceOutput)
    assert "private text" not in str(exc.value)


def test_network_error_is_safe(monkeypatch):
    monkeypatch.setattr("app.llm.adapter.requests.post", Mock(side_effect=requests.ConnectionError("secret")))
    with pytest.raises(LLMError, match="reach Claude") as exc:
        LLMClient(api_key="test", max_retries=0).chat_json("s", "u")
    assert "secret" not in str(exc.value)


def test_application_config_used():
    class Settings(Config):
        ANTHROPIC_API_KEY = "application-key"
        CLAUDE_MODEL = "application-model"
    with create_app(Settings).app_context():
        client = get_llm_client()
        assert client.configured and client.model == "application-model"
        assert client.api_key == "application-key"


def test_live_claude_pipeline_preserves_rewrite_and_fact_ids(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    tex = (root / "samples/sample_resume.tex").read_text(encoding="utf-8")
    facts = parse_resume_latex(tex)
    role = facts.experience[0]
    source = role.bullets[0]
    rewritten = source.text.replace("Responsible for designing and implementing", "Designed and implemented")
    jd_output = {field: [] for field in JobOutput.model_fields}
    jd_output.update(role_title="Software Engineer", seniority="Mid-level", requirements=[{
        "req_id": "req1", "requirement": "Python", "excerpt": "Python",
        "classification": "required", "importance": 1.0, "rationale": "required qualification", "is_core": True,
    }])
    rewrite = {
        "summary": "", "experience": [{"source_fact_id": role.fact_id,
        "employer": role.employer, "title": role.title, "functional_descriptor": "",
        "bullets": [{"text": rewritten, "fact_ids": [source.fact_id]}]}],
        "projects": [], "skills": [], "content_decisions": [], "clarification_questions": [],
    }
    post = Mock(side_effect=[message({"analysis": jd_output, "mappings": []}), message(rewrite)])
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    def fit(resume, page_size):
        return resume, "generated LaTeX", CompilationResult(error="Test compiler unavailable"), []
    monkeypatch.setattr("app.pipeline.orchestrator.fit_resume_to_one_page", fit)
    monkeypatch.setattr(Config, "MAX_OPTIMIZATION_PASSES", 1)
    result = run_optimization(tex, "Required: Python", llm=LLMClient(api_key="test"))
    assert not result.demo_mode
    assert result.job_analysis.role_title == "Software Engineer"
    assert result.optimized_resume.experience[0].bullets[0].text == rewritten
    assert result.passes_used == 1
    assert any(change.after == rewritten for change in result.change_log)
    assert post.call_count == 2
    sent = json.loads(post.call_args_list[1].kwargs["json"]["messages"][0]["content"])
    assert sent["resume_facts"]["experience"][0]["fact_id"] == role.fact_id
    assert sent["resume_facts"]["experience"][0]["bullets"][0]["fact_id"] == source.fact_id
    assert "output_schema" not in sent
    assert post.call_args_list[0].kwargs["json"]["max_tokens"] == Config.CLAUDE_PLAN_MAX_TOKENS
    assert post.call_args_list[1].kwargs["json"]["max_tokens"] == Config.CLAUDE_REWRITE_MAX_TOKENS


def test_live_failure_never_becomes_demo(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    tex = (root / "samples/sample_resume.tex").read_text(encoding="utf-8")
    monkeypatch.setattr("app.llm.adapter.requests.post", Mock(return_value=response(status=401)))
    with pytest.raises(LLMError, match="authentication"):
        run_optimization(tex, "Required: Python", llm=LLMClient(api_key="bad"))


def test_gaps_include_evidence_lost_during_fitting():
    from app.models.schemas import RequirementEvidence

    gaps = _final_gaps([
        RequirementEvidence(req_id="req1", requirement="Incident response", evidence_strength="none", current_coverage=0),
        RequirementEvidence(req_id="req2", requirement="Team leadership", evidence_strength="partial", current_coverage=0.4),
        RequirementEvidence(req_id="req3", requirement="Python", evidence_strength="direct", current_coverage=1),
    ])
    assert [gap.requirement for gap in gaps] == ["Incident response", "Team leadership"]
    assert gaps[0].gap_type == "missing" and gaps[1].gap_type == "needs_clarification"
