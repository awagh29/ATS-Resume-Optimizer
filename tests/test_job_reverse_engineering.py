"""Complete JD inventory and source-grounded enrichment, entirely offline."""

from app.models.schemas import Classification, JobAnalysis
from app.pipeline.analyze_jd import analyze_job_description, merge_llm_job_analysis


def test_required_keyword_in_first_bullet_is_content_not_a_heading():
    analysis = analyze_job_description("Engineer\nQualifications\n- Python experience is required.\n- SQL proficiency")
    assert "Python experience is required." in analysis.required_qualifications
    assert "SQL proficiency" in analysis.required_qualifications


def test_plain_paragraphs_inline_headings_and_short_tools_are_preserved():
    jd = "Required: Python; SQL; AWS\nPreferred: Rust\nResponsibilities: Build reporting APIs. Deliver reliable services."
    analysis = analyze_job_description(jd)
    assert {"Python;", "SQL;", "AWS"} <= set(analysis.required_qualifications)
    assert analysis.preferred_qualifications == ["Rust"]
    assert len(analysis.primary_responsibilities) == 2
    assert all(row.excerpt in jd for row in analysis.requirements if row.excerpt)


def test_all_constraints_survive_long_description_and_long_bullet():
    qualifications = [f"- Demonstrated capability in specialty{i}" for i in range(45)]
    long_requirement = "- " + "Professional experience in platform delivery, " * 10 + "including regulated aerospace compliance."
    analysis = analyze_job_description("Required qualifications\n" + "\n".join(qualifications + [long_requirement]))
    assert len(analysis.required_qualifications) == 46
    assert "regulated aerospace compliance." in analysis.required_qualifications[-1]
    assert any("specialty44" in row.requirement for row in analysis.requirements)
    assert any("regulated aerospace compliance." in row.excerpt for row in analysis.requirements)


def test_eligibility_and_collaboration_cover_non_bullet_content_and_stemming():
    analysis = analyze_job_description(
        "Responsibilities\nCollaborating with stakeholders and mentoring engineers.\n"
        "Required qualifications\nUS citizenship and authorization to work required.\n"
        "Preferred qualifications\nHybrid attendance preferred."
    )
    assert any("Collaborating" in item for item in analysis.leadership_collaboration)
    assert any("citizenship" in item for item in analysis.eligibility_constraints)
    hybrid = next(row for row in analysis.requirements if "Hybrid" in row.requirement)
    assert hybrid.classification == Classification.PREFERRED


def test_benefits_section_does_not_become_required_qualification():
    analysis = analyze_job_description("Requirements\nPython experience\nBenefits\nAnnual retreat and generous leave.")
    assert analysis.required_qualifications == ["Python experience"]


def test_enrichment_cannot_delete_or_downgrade_local_requirements():
    jd = "Responsibilities\nBuild Python APIs\nRequired qualifications\nCitizenship required\n5+ years of Python experience"
    base = analyze_job_description(jd)
    original = {row.requirement: row.req_id for row in base.requirements}
    merged = merge_llm_job_analysis(base, {
        "required_qualifications": ["Python"],
        "requirements": [{"req_id": "resp1", "requirement": "Build Python APIs", "excerpt": "Build Python APIs", "classification": "preferred"}],
    }, jd)
    assert {row.requirement: row.req_id for row in merged.requirements} == original
    assert next(row for row in merged.requirements if "Citizenship" in row.requirement).classification == Classification.REQUIRED
    assert "5+ years of Python experience" in merged.required_qualifications


def test_model_additions_use_unique_ids_and_reject_unrelated_or_partial_quotes():
    data = {"requirements": [
        {"req_id": "plan-extra", "requirement": "Python", "excerpt": "Python", "classification": "required"},
        {"req_id": "plan-extra", "requirement": "Python and Kubernetes", "excerpt": "Python", "classification": "required"},
        {"req_id": "years", "requirement": "5 years of Python", "excerpt": "Python", "classification": "required"},
    ]}
    merged = merge_llm_job_analysis(JobAnalysis(), data, "Required qualifications: Python")
    assert merged.requirements[0].req_id == "plan-extra"
    assert len({row.req_id for row in merged.requirements}) == 3
    assert all(row.classification == Classification.INFERRED and not row.is_core for row in merged.requirements[1:])


def test_model_cannot_promote_preference_or_general_tool_mention_to_requirement():
    data = {"requirements": [
        {"req_id": "first", "requirement": "Rust", "excerpt": "Rust", "classification": "required"},
        {"req_id": "second", "requirement": "Python", "excerpt": "Python", "classification": "required"},
    ]}
    merged = merge_llm_job_analysis(JobAnalysis(), data, "We use Python.\nPreferred: Rust")
    assert merged.requirements[0].classification == Classification.PREFERRED
    assert merged.requirements[1].classification == Classification.INFERRED


def test_unstructured_model_summary_cannot_establish_invented_requirement():
    jd = "Required: Python"
    base = analyze_job_description(jd)
    merged = merge_llm_job_analysis(base, {"required_qualifications": ["Must have Kubernetes"]}, jd)
    assert not any("Kubernetes" in row.requirement for row in merged.requirements)


def test_markdown_headings_and_source_excerpts_remain_grounded():
    jd = "## **Required qualifications**\n- Python\n**Preferred:** Rust\n**Responsibilities:** Build APIs."
    analysis = analyze_job_description(jd)
    assert analysis.required_qualifications == ["Python"]
    assert analysis.preferred_qualifications == ["Rust"]
    assert analysis.primary_responsibilities == ["Build APIs."]
    assert all(row.excerpt in jd for row in analysis.requirements)
