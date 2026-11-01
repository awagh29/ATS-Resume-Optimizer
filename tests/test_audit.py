"""Factual integrity audit tests."""

from app.models.schemas import (
    ContactInfo,
    ExperienceBullet,
    ExperienceRole,
    OptimizedBullet,
    OptimizedResume,
    OptimizedRole,
    ResumeFacts,
    SkillGroup,
    SupplementalFacts,
)
from app.pipeline.audit import audit_optimized_resume


def _facts():
    return ResumeFacts(
        contact=ContactInfo(name="Alex Rivera", email="alex@example.com"),
        experience=[
            ExperienceRole(
                fact_id="exp1",
                employer="BrightPath Labs",
                title="Software Engineer",
                start_date="2021",
                end_date="Present",
                bullets=[
                    ExperienceBullet(
                        fact_id="b1",
                        text="Built REST APIs in Python and Flask for analytics dashboards.",
                    )
                ],
            )
        ],
        skills=[SkillGroup(fact_id="sk1", category="Languages", skills=["Python", "SQL"])],
    )


def test_removes_unsupported_metric():
    facts = _facts()
    opt = OptimizedResume(
        contact=facts.contact,
        experience=[
            OptimizedRole(
                employer="BrightPath Labs",
                title="Software Engineer",
                source_fact_id="exp1",
                bullets=[
                    OptimizedBullet(
                        text="Built REST APIs in Python, resulting in 47% latency reduction.",
                        fact_ids=["b1"],
                    )
                ],
            )
        ],
        skills=facts.skills,
    )
    cleaned, result = audit_optimized_resume(opt, facts)
    assert cleaned.experience[0].bullets == []
    assert any("47%" in c or "metric" in c.lower() for c in result.removed_claims)
    assert result.clarification_questions


def test_allows_metric_when_in_supplemental():
    facts = _facts()
    supp = SupplementalFacts(metrics=["At BrightPath Labs, reduced latency by 47%"])
    opt = OptimizedResume(
        contact=facts.contact,
        experience=[
            OptimizedRole(
                employer="BrightPath Labs",
                title="Software Engineer",
                source_fact_id="exp1",
                bullets=[
                    OptimizedBullet(
                        text="Built REST APIs in Python, resulting in 47% latency reduction.",
                        fact_ids=["b1", "supplemental-1"],
                    )
                ],
            )
        ],
        skills=facts.skills,
    )
    cleaned, result = audit_optimized_resume(opt, facts, supp)
    assert len(cleaned.experience[0].bullets) == 1


def test_removes_invented_skill():
    facts = _facts()
    opt = OptimizedResume(
        contact=facts.contact,
        experience=[],
        skills=[SkillGroup(fact_id="sk1", category="Languages", skills=["Python", "Rust"])],
    )
    cleaned, _ = audit_optimized_resume(opt, facts)
    assert cleaned.skills[0].skills == ["Python"]


def test_preserves_official_title():
    facts = _facts()
    opt = OptimizedResume(
        contact=facts.contact,
        experience=[
            OptimizedRole(
                employer="BrightPath Labs",
                title="Staff Platform Architect",
                source_fact_id="exp1",
                bullets=[
                    OptimizedBullet(text="Built REST APIs in Python and Flask.", fact_ids=["b1"])
                ],
            )
        ],
    )
    cleaned, result = audit_optimized_resume(opt, facts)
    assert cleaned.experience[0].title == "Software Engineer"
    assert result.removed_claims
