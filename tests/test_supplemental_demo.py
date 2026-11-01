"""Offline optimization uses confirmed local knowledge under the factual audit."""

from app.models.schemas import (
    ExperienceBullet, ExperienceRole, JobAnalysis, PositioningChoice, ProjectItem,
    ResumeFacts, SkillGroup, SupplementalFacts,
)
from app.pipeline.audit import audit_optimized_resume
from app.pipeline.rewrite import deterministic_optimize


def _facts():
    return ResumeFacts(
        experience=[ExperienceRole(
            fact_id="role1", employer="Example Co", title="Engineer",
            bullets=[ExperienceBullet(fact_id="b1", text="Built Python APIs for analytics.")],
        )],
        projects=[ProjectItem(fact_id="project1", name="Dashboards", technologies=["Python"])],
        skills=[SkillGroup(fact_id="skills1", category="Languages", skills=["Python"])],
    )


def _optimize(facts, supplemental):
    result = deterministic_optimize(facts, JobAnalysis(), [], PositioningChoice(framing="", rationale=""), supplemental)
    optimized = result[0]
    checked, audit = audit_optimized_resume(optimized, facts, supplemental)
    assert audit.passed
    return checked, result[4]


def test_confirmed_skills_are_split_deduplicated_and_preserved_by_audit():
    facts = _facts()
    optimized, _ = _optimize(facts, SupplementalFacts(skills=["Rust; Python | SQL", "rust"]))
    skills = [skill for group in optimized.skills for skill in group.skills]
    assert set(skills) == {"Python", "Rust", "SQL"}
    assert len(skills) == 3
    assert facts.skills[0].skills == ["Python"]


def test_unique_employer_scoped_metric_is_included_with_source_citation():
    optimized, _ = _optimize(_facts(), SupplementalFacts(metrics=["At Example Co, reduced API latency by 47%."]))
    bullet = next(b for b in optimized.experience[0].bullets if "47%" in b.text)
    assert bullet.text == "reduced API latency by 47%."
    assert bullet.fact_ids == ["supplemental-1"]
    assert bullet.source_role_fact_id == "role1"


def test_unscoped_metric_does_not_gain_employer_attribution():
    optimized, decisions = _optimize(_facts(), SupplementalFacts(metrics=["Reduced API latency by 47%."]))
    assert not any("47%" in b.text for role in optimized.experience for b in role.bullets)
    assert any("attribution needs clarification" in decision for decision in decisions)


def test_same_employer_promotions_require_precise_role_id():
    facts = _facts()
    facts.experience.append(ExperienceRole(fact_id="role2", employer="Example Co", title="Senior Engineer"))
    optimized, _ = _optimize(facts, SupplementalFacts(accomplishments=["At Example Co, reduced API latency by 47%."]))
    assert not any("47%" in b.text for role in optimized.experience for b in role.bullets)


def test_precise_role_id_selects_the_correct_promotion_and_stable_supplemental_id():
    facts = _facts()
    facts.experience.append(ExperienceRole(fact_id="role2", employer="Example Co", title="Senior Engineer"))
    supplemental = SupplementalFacts(
        metrics=["Unscoped response latency improvement of 12%."],
        responsibilities=["For role2, operated Python APIs at Example Co."],
    )
    optimized, _ = _optimize(facts, supplemental)
    assert len(optimized.experience[0].bullets) == 1
    bullet = optimized.experience[1].bullets[0]
    assert bullet.fact_ids == ["supplemental-2"]
    assert bullet.text == "operated Python APIs at Example Co."


def test_confirmed_project_fact_can_restore_existing_omitted_project():
    optimized, _ = _optimize(_facts(), SupplementalFacts(accomplishments=["For Dashboards, built Python dashboards serving 20 users."]))
    assert len(optimized.projects) == 1
    assert optimized.projects[0].source_fact_id == "project1"
    assert optimized.projects[0].bullets[0].fact_ids == ["supplemental-1"]
    assert "20 users" in optimized.projects[0].bullets[0].text


def test_duplicate_project_names_cannot_assign_supplemental_fact():
    facts = _facts()
    facts.projects.append(ProjectItem(fact_id="project2", name="Dashboards"))
    optimized, _ = _optimize(facts, SupplementalFacts(accomplishments=["For Dashboards, built Python dashboards serving 20 users."]))
    assert not optimized.projects


def test_fact_that_names_multiple_existing_employers_is_left_unassigned():
    facts = _facts()
    facts.experience.append(ExperienceRole(fact_id="role2", employer="Other Co", title="Engineer"))
    optimized, _ = _optimize(facts, SupplementalFacts(accomplishments=["At Example Co and Other Co, built Python APIs serving 20 users."]))
    assert not any("20 users" in b.text for role in optimized.experience for b in role.bullets)


def test_source_ids_inside_other_ids_do_not_cross_attribute_facts():
    facts = _facts()
    facts.experience.append(ExperienceRole(fact_id="role10", employer="Other Co", title="Engineer"))
    optimized, _ = _optimize(facts, SupplementalFacts(accomplishments=["For role10, built Python APIs serving 20 users."]))
    assert len(optimized.experience[0].bullets) == 1
    assert optimized.experience[1].bullets[0].fact_ids == ["supplemental-1"]


def test_duplicate_confirmed_original_bullet_does_not_duplicate_resume_content():
    optimized, _ = _optimize(_facts(), SupplementalFacts(accomplishments=["At Example Co, Built Python APIs for analytics."]))
    assert len(optimized.experience[0].bullets) == 1
