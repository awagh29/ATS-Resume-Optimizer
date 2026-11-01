"""Knowledge-backed optimization integration with no model or compiler calls."""

import json
from unittest.mock import Mock

import pytest

from app.config import Config
from app.knowledge import KnowledgeStore
from app.llm.adapter import LLMClient
from app.llm.outputs import JobOutput
from app.models.schemas import CompilationResult, OptimizationResult
from app.pipeline.orchestrator import run_optimization
from app.pipeline.parse_resume import parse_resume_latex
from app.pipeline.scoring import optimized_to_text


_REAL_CHAT_JSON = LLMClient.chat_json


RESUME = r"""\begin{document}
Alex Example
\section{Experience}
\resumeSubheading{BrightPath Labs}{2021 -- Present}{Software Engineer}{Seattle, WA}
\resumeItem{Built Python REST APIs for analytics dashboards.}
\section{Skills}
\textbf{Languages:} Python
\end{document}"""


@pytest.fixture(autouse=True)
def local_only_pipeline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Knowledge integration tests must never call a model")

    def fit(optimized, *, page_size):
        latex = r"\documentclass{article}\begin{document}" + optimized_to_text(optimized) + r"\end{document}"
        return optimized, latex, CompilationResult(
            success=True, compiler_available=True, page_count=1, layout_verified=True,
            pdf_bytes=b"%PDF-offline-fixture",
        ), []

    monkeypatch.setattr(LLMClient, "chat_json", forbidden)
    monkeypatch.setattr("app.pipeline.orchestrator.fit_resume_to_one_page", fit)


@pytest.fixture
def store():
    value = KnowledgeStore(":memory:")
    yield value
    value.close()


def optimize(store, jd, **kwargs):
    return run_optimization(RESUME, jd, llm=LLMClient(api_key=""),
                            knowledge_store=store, **kwargs)


def test_retrieved_confirmed_skill_is_used_and_can_be_opted_out(store):
    entry = store.add_entry("Rust skill", "Rust", category="skill")
    jd = "Backend Engineer\nRequired qualifications:\n- Rust development experience.\n- Python REST APIs."
    with_knowledge = optimize(store, jd)
    without_knowledge = optimize(store, jd, use_knowledge=False)
    assert entry["id"] in {item["id"] for item in with_knowledge.knowledge_used}
    assert "Rust" in [skill for group in with_knowledge.optimized_resume.skills for skill in group.skills]
    assert without_knowledge.knowledge_used == []
    assert "Rust" not in [skill for group in without_knowledge.optimized_resume.skills for skill in group.skills]
    assert with_knowledge.token_usage["api_calls"] == 0
    assert with_knowledge.token_usage["input_tokens"] == 0
    assert with_knowledge.token_usage["output_tokens"] == 0


def test_scoped_accomplishment_has_durable_source_mapping(store):
    content = "At BrightPath Labs, reduced REST API latency by 47%."
    entry = store.add_entry("API latency improvement", content, category="accomplishment")
    result = optimize(store, "Backend Engineer\nResponsibilities:\n- Reduce REST API latency.")
    source = next(item for item in result.knowledge_used if item["id"] == entry["id"])
    assert source["content"] == content
    assert source["fact_ids"]
    bullets = result.optimized_resume.experience[0].bullets
    assert any("47%" in bullet.text for bullet in bullets)
    assert any(set(source["fact_ids"]) & set(bullet.fact_ids) for bullet in bullets if "47%" in bullet.text)
    stored = store.get_run(result.run_id)["result"]
    assert stored["knowledge_used"] == result.knowledge_used


def test_generated_history_never_supplies_candidate_qualifications(store):
    fabricated = OptimizationResult()
    fabricated.optimized_resume.summary = "Managed Kubernetes clusters for 900000 users."
    store.save_run(fabricated.to_template_dict(), job_description="Kubernetes platform engineering")
    result = optimize(store, "Platform Engineer\nRequired qualifications:\n- Kubernetes cluster management.")
    assert result.knowledge_used == []
    assert "Kubernetes" not in optimized_to_text(result.optimized_resume)
    assert result.target_reached is False
    assert any("Kubernetes" in gap.requirement for gap in result.gaps)


def test_completed_run_survives_store_reopen_with_editable_output(tmp_path):
    path = tmp_path / "local.sqlite3"
    store = KnowledgeStore(path)
    result = optimize(store, "Backend Engineer\nRequired qualifications:\n- Python REST API development.",
                      page_size="a4", company="BrightPath")
    assert result.run_id
    store.close()
    reopened = KnowledgeStore(path)
    try:
        run = reopened.get_run(result.run_id)
        saved = run["result"]
        assert run["company"] == "BrightPath"
        assert run["resume_hash"]
        assert saved["run_id"] == result.run_id
        assert saved["page_size"] == "a4"
        assert saved["token_usage"] == result.token_usage
        assert saved["latex_source"] == result.latex_source
        assert saved["compilation"].get("pdf_bytes") is None
        assert saved["demo_mode"] is True
    finally:
        reopened.close()


def test_deleting_knowledge_excludes_it_from_future_runs_but_keeps_history(store):
    entry = store.add_entry("Rust skill", "Rust", category="skill")
    jd = "Backend Engineer\nRequired qualifications:\n- Rust development experience."
    first = optimize(store, jd)
    assert first.knowledge_used
    assert store.delete_entry(entry["id"])
    second = optimize(store, jd)
    assert second.knowledge_used == []
    assert "Rust" not in [skill for group in second.optimized_resume.skills for skill in group.skills]
    assert store.get_run(first.run_id)["result"]["knowledge_used"]


def test_candidate_profiles_do_not_share_evidence_or_history(store):
    entry = store.add_entry("Rust skill", "Rust", category="skill", profile_id="candidate-a")
    result = optimize(store, "Engineer\nRequired qualifications:\n- Rust experience.", profile_id="candidate-b")
    assert result.knowledge_used == []
    assert store.get_run(result.run_id, profile_id="candidate-b")
    assert store.get_run(result.run_id, profile_id="candidate-a") is None
    assert store.list_entries(profile_id="candidate-a")[0]["id"] == entry["id"]


def test_durable_response_cache_reuses_pipeline_output_and_invalidates_changed_jd(tmp_path, monkeypatch):
    """Exercise the real adapter through mocked HTTP and reopen both client/store."""
    monkeypatch.setattr(Config, "MAX_OPTIMIZATION_PASSES", 1)
    monkeypatch.setattr(LLMClient, "chat_json", _REAL_CHAT_JSON)
    facts = parse_resume_latex(RESUME)
    role = facts.experience[0]
    source = role.bullets[0]
    rewritten = "Designed Python REST APIs for analytics dashboards."
    analysis = {field: [] for field in JobOutput.model_fields}
    analysis.update(role_title="Software Engineer", seniority="Mid-level",
                    required_qualifications=["Python"], tools_and_tech=["Python"])
    changed_analysis = dict(analysis, required_qualifications=["Python", "SQL"],
                            tools_and_tech=["Python", "SQL"])
    rewrite = {
        "summary": "", "experience": [{
            "source_fact_id": role.fact_id, "employer": role.employer, "title": role.title,
            "functional_descriptor": "", "bullets": [{"text": rewritten, "fact_ids": [source.fact_id]}],
        }], "projects": [], "skills": [], "content_decisions": [], "clarification_questions": [],
    }

    def message(payload):
        return Mock(status_code=200, headers={}, json=Mock(return_value={
            "stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(payload)}],
            "usage": {"input_tokens": 100, "output_tokens": 30},
        }))

    post = Mock(side_effect=[
        message({"analysis": analysis, "mappings": []}), message(rewrite),
        message({"analysis": changed_analysis, "mappings": []}), message(rewrite),
    ])
    monkeypatch.setattr("app.llm.adapter.requests.post", post)
    path = tmp_path / "cached-pipeline.sqlite3"
    jd = "Software Engineer\nRequired qualifications:\n- Python REST API development."

    def run(store, description):
        return run_optimization(
            RESUME, description, llm=LLMClient(api_key="mock-only-key", model="claude-sonnet-5-5", max_retries=0),
            knowledge_store=store,
        )

    first_store = KnowledgeStore(path)
    try:
        first = run(first_store, jd)
        assert post.call_count == 2
        assert first.token_usage["api_calls"] == 2
        assert first.token_usage["local_cache_hits"] == 0
        assert first.token_usage["input_tokens"] == 200
        assert first.optimized_resume.experience[0].bullets[0].text == rewritten
    finally:
        first_store.close()

    reopened_store = KnowledgeStore(path)
    try:
        repeated = run(reopened_store, jd)
        assert post.call_count == 2
        assert repeated.token_usage["api_calls"] == 0
        assert repeated.token_usage["local_cache_hits"] == 2
        assert repeated.token_usage["input_tokens"] == 0
        assert repeated.token_usage["output_tokens"] == 0
        assert repeated.optimized_resume.experience[0].bullets[0].text == rewritten
        assert repeated.optimized_resume.experience[0].bullets[0].fact_ids == [source.fact_id]
        assert len(reopened_store.list_runs()) == 2
        assert reopened_store.get_run(repeated.run_id)["result"]["token_usage"]["api_calls"] == 0

        changed = run(reopened_store, jd + "\n- SQL database experience.")
        assert post.call_count == 4
        assert changed.token_usage["api_calls"] == 2
        assert changed.token_usage["local_cache_hits"] == 0
        assert len(reopened_store.list_runs()) == 3
    finally:
        reopened_store.close()
