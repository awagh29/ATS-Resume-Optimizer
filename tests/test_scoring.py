"""Scoring consistency tests."""

from app.models.schemas import (
    Classification,
    EvidenceStrength,
    RequirementEvidence,
)
from app.pipeline.scoring import score_from_matrix


def _matrix():
    return [
        RequirementEvidence(
            req_id="req1",
            requirement="Strong proficiency in Python",
            classification=Classification.REQUIRED,
            importance=0.95,
            matching_evidence="Languages: Python",
            evidence_strength=EvidenceStrength.DIRECT,
            current_coverage=1.0,
        ),
        RequirementEvidence(
            req_id="req2",
            requirement="Experience with Kubernetes",
            classification=Classification.REQUIRED,
            importance=0.9,
            matching_evidence="",
            evidence_strength=EvidenceStrength.NONE,
            current_coverage=0.0,
        ),
        RequirementEvidence(
            req_id="resp1",
            requirement="Build REST APIs",
            classification=Classification.REQUIRED,
            importance=0.85,
            rationale="primary responsibility",
            matching_evidence="Implemented REST APIs in Python",
            evidence_strength=EvidenceStrength.DIRECT,
            current_coverage=1.0,
        ),
        RequirementEvidence(
            req_id="pref1",
            requirement="Nice to have GraphQL",
            classification=Classification.PREFERRED,
            importance=0.4,
            matching_evidence="",
            evidence_strength=EvidenceStrength.NONE,
            current_coverage=0.0,
        ),
    ]


def test_score_bounds_and_recompute():
    text = "Python engineer. Implemented REST APIs in Python and SQL. Languages: Python, SQL."
    score = score_from_matrix(
        _matrix(),
        text,
        tools_and_tech=["Python", "SQL", "Kubernetes"],
        domain_vocabulary=["REST"],
    )
    assert 0 <= score.total <= 100
    assert score.required_qualification_evidence <= 35
    assert score.primary_responsibility_coverage <= 25
    assert score.skills_and_terminology <= 15
    assert score.evidence_and_accomplishment_quality <= 15
    assert score.ats_readability_and_structure <= 10
    parts = (
        score.required_qualification_evidence
        + score.primary_responsibility_coverage
        + score.skills_and_terminology
        + score.evidence_and_accomplishment_quality
        + score.ats_readability_and_structure
    )
    assert abs(parts - score.total) < 0.15


def test_mandatory_gaps_listed_even_with_preferred_elsewhere():
    text = "Python REST APIs SQL dashboard GraphQL GraphQL GraphQL"
    score = score_from_matrix(
        _matrix(),
        text,
        tools_and_tech=["Python", "Kubernetes", "GraphQL"],
    )
    assert any("Kubernetes" in g for g in score.mandatory_gaps)


def test_same_matrix_same_inputs_stable():
    text = "Built REST APIs using Python and PostgreSQL on AWS with Docker."
    a = score_from_matrix(_matrix(), text, tools_and_tech=["Python", "AWS", "Docker"])
    b = score_from_matrix(_matrix(), text, tools_and_tech=["Python", "AWS", "Docker"])
    assert a.total == b.total
    assert a.required_qualification_evidence == b.required_qualification_evidence


def test_keyword_stuffing_not_rewarded_unbounded():
    base = "Python SQL REST"
    stuffed = "Python SQL REST " + ("Kubernetes " * 50)
    matrix = _matrix()
    # Neither has k8s evidence in matrix coverage for req2
    s1 = score_from_matrix(matrix, base, tools_and_tech=["Python", "SQL", "Kubernetes"])
    s2 = score_from_matrix(matrix, stuffed, tools_and_tech=["Python", "SQL", "Kubernetes"])
    # Skills term hit for k8s may increase skills bucket, but required gaps remain
    assert any("Kubernetes" in g for g in s2.mandatory_gaps)
    assert s2.skills_and_terminology <= 15
