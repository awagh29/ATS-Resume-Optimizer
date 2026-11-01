"""Regression coverage for source extraction and evidence provenance."""

from pathlib import Path

from app.models.schemas import (
    Classification, ContactInfo, ExperienceBullet, ExperienceRole, JobAnalysis,
    OptimizedBullet, OptimizedResume, OptimizedRole, RequirementEvidence,
    RequirementItem, ResumeFacts, SkillGroup, SupplementalFacts, EvidenceStrength,
    parse_supplemental_text,
)
from app.pipeline.analyze_jd import analyze_job_description, merge_llm_job_analysis
from app.pipeline.audit import audit_optimized_resume
from app.pipeline.evidence_matrix import build_evidence_matrix, rematch_matrix_for_text
from app.pipeline.parse_resume import parse_resume_latex
from app.pipeline.rewrite import merge_llm_optimized
from app.pipeline.scoring import optimized_to_text


def _facts():
    facts = ResumeFacts(
        contact=ContactInfo(name="Alex Example"),
        experience=[
            ExperienceRole(fact_id="role1", employer="Example Co", title="Engineer", start_date="2020", end_date="2022", bullets=[ExperienceBullet(fact_id="b1", text="Built Python APIs for analytics dashboards.")]),
            ExperienceRole(fact_id="role2", employer="Example Co", title="Senior Engineer", start_date="2022", end_date="Present", bullets=[ExperienceBullet(fact_id="b2", text="Built Java services and reduced latency by 20%.")]),
        ],
        skills=[SkillGroup(fact_id="sk1", category="Languages", skills=["Python", "Java"])],
    )
    facts.collect_fact_ids()
    return facts


def _optimized(facts, text, ids=None):
    return OptimizedResume(experience=[OptimizedRole(employer="Example Co", title="Engineer", source_fact_id="role1", bullets=[OptimizedBullet(text=text, fact_ids=ids or ["b1"])])])


def test_sample_extraction_preserves_identity_employers_and_sections():
    sample = Path(__file__).parents[1] / "samples" / "sample_resume.tex"
    facts = parse_resume_latex(sample.read_text(encoding="utf-8"))
    assert facts.contact.name == "Alex Rivera"
    assert [(role.employer, role.title, len(role.bullets)) for role in facts.experience] == [("BrightPath Labs", "Software Engineer", 5), ("Harbor Retail", "Backend Developer Intern", 2)]
    assert len(facts.projects) == 1 and len(facts.projects[0].bullets) == 2
    assert len(facts.education) == 1
    assert facts.education[0].degree == "B.S. in Computer Science"
    assert "B.S. in Computer Science" not in facts.education[0].details
    assert "AWS (ECS, S3, CloudWatch)" in [s for g in facts.skills for s in g.skills]


def test_wrapped_bullet_keeps_verified_outcome():
    facts = parse_resume_latex(r"""\section{Experience}
Engineer at Example Co 2020 -- Present
\begin{itemize}
\item Built Python APIs
that reduced latency by 20\%.
\end{itemize}""")
    assert facts.experience[0].bullets[0].text == "Built Python APIs that reduced latency by 20%."
    assert facts.contact.name == ""


def test_common_custom_macros_preserve_role_data():
    facts = parse_resume_latex(r"""\begin{document}
Alex Example
\section{Experience}
\resumeSubheading{Example Co}{2020 -- Present}{Software Engineer}{Seattle, WA}
\resumeItem{Built Python APIs for analytics.}
\end{document}""")
    assert facts.experience[0].employer == "Example Co"
    assert facts.experience[0].title == "Software Engineer"
    assert facts.experience[0].bullets[0].text == "Built Python APIs for analytics."


def test_nested_formatting_in_custom_macro_keeps_data():
    facts = parse_resume_latex(r"""\begin{document}
\section{Experience}
\resumeSubheading{\textbf{Example Co}}{2020 -- Present}{Software Engineer}{Seattle, WA}
\resumeItem{Built \textbf{Python} APIs for analytics.}
\end{document}""")
    assert facts.experience[0].employer == "Example Co"
    assert facts.experience[0].bullets[0].text == "Built Python APIs for analytics."


def test_moderncv_cventry_preserves_role_and_bullet():
    facts = parse_resume_latex(r"""\begin{document}
\section{Experience}
\cventry{2020 -- Present}{Software Engineer}{Example Co}{Seattle, WA}{}{Built Python APIs.}
\end{document}""")
    assert facts.experience[0].employer == "Example Co"
    assert facts.experience[0].title == "Software Engineer"
    assert facts.experience[0].bullets[0].text == "Built Python APIs."


def test_same_employer_roles_merge_by_id_and_keep_omitted_history():
    facts = _facts()
    fallback = OptimizedResume(experience=[OptimizedRole(employer=r.employer, title=r.title, source_fact_id=r.fact_id) for r in facts.experience])
    merged = merge_llm_optimized(fallback, {"experience": [{"source_fact_id": "role1", "employer": "Example Co", "title": "Invented title", "bullets": [{"text": "Built Python APIs for analytics.", "fact_ids": ["b1"]}]}]}, facts)
    assert [role.title for role in merged.experience] == ["Engineer", "Senior Engineer"]
    assert merged.experience[0].source_fact_id == "role1"


def test_cited_id_does_not_allow_invented_tool_or_unrelated_claim():
    facts = _facts()
    for text in ("Built Python APIs using Kubernetes.", "Secured classified orbital launch infrastructure."):
        cleaned, audit = audit_optimized_resume(_optimized(facts, text), facts)
        assert not cleaned.experience[0].bullets
        assert not audit.passed


def test_metric_from_different_role_or_date_cannot_support_claim():
    facts = _facts()
    cleaned, audit = audit_optimized_resume(_optimized(facts, "Built Python APIs and reduced latency by 20%."), facts)
    assert not cleaned.experience[0].bullets
    assert any("20%" in claim for claim in audit.removed_claims)


def test_unscoped_supplemental_metric_cannot_be_assigned_to_employer():
    facts = _facts()
    optimized = _optimized(facts, "Built Python APIs and reduced latency by 47%.", ["b1", "supplemental-1"])
    cleaned, _ = audit_optimized_resume(optimized, facts, SupplementalFacts(metrics=["Reduced latency by 47%."]))
    assert not cleaned.experience[0].bullets


def test_supplemental_metric_for_other_role_cannot_cross_employers():
    facts = _facts()
    optimized = _optimized(facts, "Built Python APIs and reduced latency by 47%.", ["b1", "supplemental-1"])
    cleaned, _ = audit_optimized_resume(optimized, facts, SupplementalFacts(metrics=["For role2, reduced latency by 47%."]))
    assert not cleaned.experience[0].bullets


def test_scoped_supplemental_metric_is_preserved_through_merge_and_audit():
    facts = _facts()
    supplemental = SupplementalFacts(metrics=["For role1, reduced latency by 47%."])
    fallback = _optimized(facts, "Built Python APIs for analytics dashboards.")
    merged = merge_llm_optimized(fallback, {"experience": [{"source_fact_id": "role1", "bullets": [{"text": "Built Python APIs and reduced latency by 47%.", "fact_ids": ["b1", "supplemental-1"]}]}]}, facts, supplemental)
    cleaned, _ = audit_optimized_resume(merged, facts, supplemental)
    assert cleaned.experience[0].bullets[0].fact_ids == ["b1", "supplemental-1"]


def test_confirmed_unscoped_skill_can_enter_skills_section():
    facts = _facts()
    supplemental = SupplementalFacts(skills=["Rust"])
    merged = merge_llm_optimized(OptimizedResume(skills=facts.skills), {"skills": [{"category": "Languages", "skills": ["Python", "Rust"]}]}, facts, supplemental)
    cleaned, _ = audit_optimized_resume(merged, facts, supplemental)
    assert cleaned.skills[0].skills == ["Python", "Rust"]


def test_unknown_or_cross_role_source_id_rejected():
    facts = _facts()
    cleaned, _ = audit_optimized_resume(_optimized(facts, "Built Python APIs for analytics dashboards.", ["b2"]), facts)
    assert not cleaned.experience[0].bullets


def test_hallucinated_evidence_text_is_not_accepted_from_valid_ids():
    facts = _facts()
    analysis = JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="Kubernetes orchestration")])
    matrix = build_evidence_matrix(analysis, facts, llm_mappings=[{"req_id": "req1", "evidence_strength": "direct", "evidence_fact_ids": ["b1"], "matching_evidence": "Managed Kubernetes orchestration clusters."}])
    assert matrix[0].evidence_strength == EvidenceStrength.NONE


def test_real_but_unrelated_quote_cannot_establish_qualification():
    facts = _facts()
    analysis = JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="Bayesian inference expertise")])
    matrix = build_evidence_matrix(analysis, facts, llm_mappings=[{"req_id": "req1", "evidence_strength": "direct", "evidence_fact_ids": ["b1"], "matching_evidence": facts.experience[0].bullets[0].text}])
    assert matrix[0].evidence_strength == EvidenceStrength.NONE


def test_evidence_relevance_accepts_established_multiword_aliases():
    facts = _facts()
    facts.experience[0].bullets[0].text = "Built ML models."
    analysis = JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="Machine learning")])
    matrix = build_evidence_matrix(analysis, facts, llm_mappings=[{"req_id": "req1", "evidence_strength": "direct", "evidence_fact_ids": ["b1"], "matching_evidence": "Built ML models."}])
    assert matrix[0].evidence_strength == EvidenceStrength.DIRECT


def test_invalid_citation_does_not_become_generic_role_grounding():
    facts = _facts()
    original = "Built Python APIs for analytics dashboards."
    fallback = _optimized(facts, original)
    merged = merge_llm_optimized(fallback, {"experience": [{"source_fact_id": "role1", "bullets": [{"text": "Built Python APIs using Kubernetes.", "fact_ids": ["invented-id"]}]}]}, facts)
    assert merged.experience[0].bullets[0].text == original
    assert "Kubernetes" not in merged.experience[0].bullets[0].text


def test_supplemental_evidence_has_stable_id():
    facts = _facts()
    analysis = JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="Kubernetes orchestration")])
    matrix = build_evidence_matrix(analysis, facts, SupplementalFacts(accomplishments=["Managed Kubernetes orchestration clusters."]))
    assert matrix[0].evidence_fact_ids == ["supplemental-1"]


def test_one_skill_does_not_establish_tenure_or_multiple_tools():
    facts = _facts()
    analysis = JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="5 years of Python experience"), RequirementItem(req_id="req2", requirement="Python and Kubernetes experience")])
    matrix = build_evidence_matrix(analysis, facts)
    assert all(row.current_coverage <= 0.4 for row in matrix)


def test_rematch_cannot_turn_missing_qualification_into_evidence():
    matrix = [RequirementEvidence(req_id="req1", requirement="Experience with Kubernetes orchestration", evidence_strength=EvidenceStrength.NONE, current_coverage=0)]
    assert rematch_matrix_for_text(matrix, "Experience with Kubernetes orchestration")[0].current_coverage == 0


def test_retained_granular_fact_keeps_coverage_through_paraphrase():
    facts = _facts()
    optimized = _optimized(facts, "Shipped reporting endpoints for customers.")
    matrix = [RequirementEvidence(req_id="req1", requirement="Build analytics dashboards", matching_evidence=facts.experience[0].bullets[0].text, evidence_fact_ids=["b1", "role1"], evidence_strength=EvidenceStrength.DIRECT, current_coverage=1)]
    final = rematch_matrix_for_text(matrix, optimized_to_text(optimized), facts=facts, optimized=optimized)
    assert final[0].current_coverage == 1
    assert final[0].evidence_strength == EvidenceStrength.DIRECT


def test_omitted_bullet_loses_coverage_even_when_role_is_retained():
    facts = _facts()
    facts.experience[0].bullets.append(ExperienceBullet(fact_id="b3", text="Coordinated releases."))
    optimized = _optimized(facts, "Coordinated releases.", ["b3"])
    matrix = [RequirementEvidence(req_id="req1", requirement="Build analytics dashboards", matching_evidence=facts.experience[0].bullets[0].text, evidence_fact_ids=["b1", "role1"], evidence_strength=EvidenceStrength.DIRECT, current_coverage=1)]
    final = rematch_matrix_for_text(matrix, optimized_to_text(optimized), facts=facts, optimized=optimized)
    assert final[0].current_coverage == 0


def test_parent_role_citation_does_not_retain_an_omitted_quoted_bullet():
    facts = _facts()
    optimized = OptimizedResume(experience=[OptimizedRole(employer="Example Co", title="Engineer", source_fact_id="role1")])
    matrix = [RequirementEvidence(req_id="req1", requirement="Build analytics dashboards", matching_evidence=facts.experience[0].bullets[0].text, evidence_fact_ids=["role1"], evidence_strength=EvidenceStrength.DIRECT, current_coverage=1)]
    final = rematch_matrix_for_text(matrix, optimized_to_text(optimized), facts=facts, optimized=optimized)
    assert final[0].current_coverage == 0


def test_python_qualifiers_do_not_create_false_gap_in_retained_skills():
    facts = _facts()
    optimized = OptimizedResume(skills=[SkillGroup(fact_id="new-group", category="Languages", skills=["Python"])])
    analysis = JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="Strong proficiency in Python")])
    matrix = build_evidence_matrix(analysis, facts)
    final = rematch_matrix_for_text(matrix, optimized_to_text(optimized), facts=facts, optimized=optimized)
    assert final[0].current_coverage == matrix[0].current_coverage >= 0.85


def test_retained_skill_group_does_not_retain_removed_python_skill():
    facts = _facts()
    optimized = OptimizedResume(skills=[SkillGroup(fact_id="sk1", category="Languages", skills=["Java"])])
    analysis = JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="Strong proficiency in Python")])
    matrix = build_evidence_matrix(analysis, facts)
    final = rematch_matrix_for_text(matrix, optimized_to_text(optimized), facts=facts, optimized=optimized)
    assert final[0].current_coverage == 0


def test_ci_checks_and_github_actions_support_transferable_ci_cd_evidence():
    facts = _facts()
    facts.experience[0].bullets[0].text = "Wrote integration tests and improved CI checks in GitHub Actions."
    matrix = build_evidence_matrix(JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="CI/CD experience")]), facts)
    assert matrix[0].evidence_strength == EvidenceStrength.TRANSFERABLE
    assert matrix[0].current_coverage == 0.7


def test_mentored_is_evidence_of_mentorship():
    facts = _facts()
    facts.experience[0].bullets[0].text = "Mentored one junior engineer on testing practices."
    matrix = build_evidence_matrix(JobAnalysis(requirements=[RequirementItem(req_id="req1", requirement="Mentorship experience")]), facts)
    assert matrix[0].current_coverage >= 0.85


def test_multiline_supplemental_input_cannot_transfer_metrics_via_raw_blob():
    facts = _facts()
    facts.experience[1].employer = "Other Co"
    supplemental = parse_supplemental_text("Metric: At Other Co, reduced latency by 47%.\nAccomplishment: At Example Co, supported Python APIs.")
    assert all(not ("Other Co" in line and "Example Co" in line) for line in supplemental.confirmed_lines())
    optimized = _optimized(facts, "Built Python APIs and reduced latency by 47%.", ["b1", "supplemental-1"])
    cleaned, _ = audit_optimized_resume(optimized, facts, supplemental)
    assert not cleaned.experience[0].bullets


def test_tech_matching_uses_boundaries_and_mentions_are_not_required():
    analysis = analyze_job_description("Engineer\nWe use JavaScript for ongoing development.\nNice to have:\n- Python")
    assert "JavaScript" in analysis.tools_and_tech
    assert "Java" not in analysis.tools_and_tech
    assert "Go" not in analysis.tools_and_tech
    assert not any(row.classification == Classification.REQUIRED and row.requirement == "Experience with JavaScript" for row in analysis.requirements)


def test_claude_requirement_citations_are_verified_and_ids_are_unique():
    data = {"requirements": [
        {"req_id": "duplicate", "requirement": "Python", "excerpt": "Experience with Python", "classification": "required"},
        {"req_id": "duplicate", "requirement": "Kubernetes", "excerpt": "Must know Kubernetes", "classification": "required"},
    ]}
    analysis = merge_llm_job_analysis(JobAnalysis(), data, "Experience with Python is required.")
    assert len({row.req_id for row in analysis.requirements}) == 2
    assert analysis.requirements[1].classification == Classification.INFERRED
