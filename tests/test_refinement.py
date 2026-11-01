"""A bounded refinement must retain the best audited, fitted resume."""

from pathlib import Path

import pytest

from app.config import Config
from app.llm.adapter import LLMError
from app.llm.outputs import JobOutput
from app.models.schemas import CompilationResult, MatchAssessment, ScoreBreakdown
from app.pipeline.orchestrator import run_optimization
from app.pipeline.parse_resume import parse_resume_latex


def setup_pipeline(monkeypatch, scores, *, gaps=None, refine_error=False):
    tex = (Path(__file__).parents[1] / "samples/sample_resume.tex").read_text(encoding="utf-8")
    facts = parse_resume_latex(tex)
    role = facts.experience[0]
    plan = {field: [] for field in JobOutput.model_fields}
    plan.update(role_title="Engineer", seniority="", requirements=[])
    rewrite = {"summary": "", "experience": [{"source_fact_id": role.fact_id,
               "employer": role.employer, "title": role.title, "functional_descriptor": "",
               "bullets": [{"text": role.bullets[0].text, "fact_ids": [role.bullets[0].fact_id]}]}],
               "projects": [], "skills": [], "content_decisions": [], "clarification_questions": []}

    class FakeClient:
        configured = True
        model = "fake-sonnet"
        calls = 0

        def chat_json(self, system, user, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"analysis": plan, "mappings": []}
            if self.calls == 3 and refine_error:
                raise LLMError("Test refinement error")
            if self.calls > 3:
                pytest.fail("More than one refinement requested")
            return rewrite

    assessments = iter(scores)
    counter = {"fit": 0, "score": 0}

    def assess(matrix, *args, **kwargs):
        counter["score"] += 1
        mandatory = gaps if counter["score"] == 3 else []
        return MatchAssessment(matrix=matrix, score=ScoreBreakdown(total=next(assessments),
                                                                 mandatory_gaps=mandatory or []))

    def fit(resume, **kwargs):
        counter["fit"] += 1
        return resume, f"latex-{counter['fit']}", CompilationResult(
            success=True, layout_verified=True, page_count=1, pdf_bytes=b"%PDF-stub"), []

    monkeypatch.setattr("app.pipeline.orchestrator.build_assessment", assess)
    monkeypatch.setattr("app.pipeline.orchestrator.fit_resume_to_one_page", fit)
    monkeypatch.setattr(Config, "MAX_OPTIMIZATION_PASSES", 2)
    client = FakeClient()
    return tex, client


def test_accepts_better_refinement_and_stops_at_two_passes(monkeypatch):
    tex, client = setup_pipeline(monkeypatch, [50, 90, 96])
    result = run_optimization(tex, "Required: Python", llm=client)
    assert result.latex_source == "latex-2"
    assert result.optimized_assessment.score.total == 96
    assert result.target_reached and result.passes_used == 2
    assert client.calls == 3


@pytest.mark.parametrize("score,gaps", [(85, []), (98, ["Unverified mandatory credential"])])
def test_rejects_worse_or_new_mandatory_gap_refinement(monkeypatch, score, gaps):
    tex, client = setup_pipeline(monkeypatch, [50, 90, score], gaps=gaps)
    result = run_optimization(tex, "Required: Python", llm=client)
    assert result.latex_source == "latex-1"
    assert result.optimized_assessment.score.total == 90
    assert not result.target_reached and client.calls == 3


def test_refinement_failure_keeps_first_result(monkeypatch):
    tex, client = setup_pipeline(monkeypatch, [50, 90], refine_error=True)
    result = run_optimization(tex, "Required: Python", llm=client)
    assert result.latex_source == "latex-1"
    assert any("optional refinement" in warning for warning in result.warnings)


def test_target_score_skips_refinement(monkeypatch):
    tex, client = setup_pipeline(monkeypatch, [50, 96])
    result = run_optimization(tex, "Required: Python", llm=client)
    assert result.target_reached and result.passes_used == 1 and client.calls == 2
