"""End-to-end demo-mode pipeline on sample fixtures."""

from pathlib import Path

from app.llm.adapter import LLMClient
from app.pipeline.orchestrator import run_optimization
from app.pipeline.parse_resume import parse_resume_latex
from app.pipeline.analyze_jd import analyze_job_description

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def test_parse_sample_resume():
    tex = (SAMPLES / "sample_resume.tex").read_text(encoding="utf-8")
    facts = parse_resume_latex(tex)
    assert facts.contact.email
    assert facts.experience
    assert any("Python" in s for sg in facts.skills for s in sg.skills)


def test_analyze_sample_jd():
    jd = (SAMPLES / "sample_job_description.txt").read_text(encoding="utf-8")
    analysis = analyze_job_description(jd, company="Northwind Analytics", role_title="")
    assert analysis.role_title
    assert analysis.required_qualifications or analysis.requirements
    assert analysis.tools_and_tech


def test_demo_optimization_runs():
    tex = (SAMPLES / "sample_resume.tex").read_text(encoding="utf-8")
    jd = (SAMPLES / "sample_job_description.txt").read_text(encoding="utf-8")
    # Force demo path with empty API key
    client = LLMClient(api_key="")
    result = run_optimization(
        tex,
        jd,
        company="Northwind Analytics",
        role_title="Senior Backend Engineer",
        page_size="letter",
        llm=client,
    )
    assert result.demo_mode is True
    assert "DEMO MODE" in result.demo_notice
    assert result.latex_source
    assert "\\documentclass" in result.latex_source
    assert result.optimized_resume.experience
    # Official title preserved
    assert result.optimized_resume.experience[0].title == "Software Engineer"
    # Scores present and comparable rubric
    assert 0 <= result.original_assessment.score.total <= 100
    assert 0 <= result.optimized_assessment.score.total <= 100
    # Should not invent Kubernetes as a skill
    all_skills = [s for sg in result.optimized_resume.skills for s in sg.skills]
    assert "Kubernetes" not in all_skills
    # Gaps should mention something unsupported like years or k8s / authorization ideally
    assert result.gaps or result.optimized_assessment.score.mandatory_gaps


def test_does_not_claim_95_without_evidence():
    tex = r"""
    \section*{Experience}
    Intern \hfill 2024
    \begin{itemize}
    \item Helped with paperwork.
    \end{itemize}
    """
    jd = (SAMPLES / "sample_job_description.txt").read_text(encoding="utf-8")
    result = run_optimization(tex, jd, llm=LLMClient(api_key=""))
    assert result.optimized_assessment.score.total < 95
    assert result.target_reached is False
    assert result.score_ceiling_reason
