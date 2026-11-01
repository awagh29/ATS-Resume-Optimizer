"""Typed contracts for Claude's native JSON structured outputs."""

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import Classification, EvidenceStrength


class StructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class JobRequirement(StructuredOutput):
    req_id: str
    requirement: str
    excerpt: str
    classification: Classification
    importance: float = Field(ge=0, le=1)
    rationale: str
    is_core: bool


class JobOutput(StructuredOutput):
    role_title: str
    seniority: str
    primary_responsibilities: list[str]
    required_qualifications: list[str]
    preferred_qualifications: list[str]
    tools_and_tech: list[str]
    domain_vocabulary: list[str]
    deliverables_and_outcomes: list[str]
    leadership_collaboration: list[str]
    eligibility_constraints: list[str]
    repeated_themes: list[str]
    hiring_priorities: list[str]
    inferred_notes: list[str]
    requirements: list[JobRequirement]


class EvidenceMapping(StructuredOutput):
    req_id: str
    matching_evidence: str
    evidence_fact_ids: list[str]
    evidence_strength: EvidenceStrength
    recommended_treatment: str


class EvidenceOutput(StructuredOutput):
    mappings: list[EvidenceMapping]


class PlanOutput(StructuredOutput):
    analysis: JobOutput
    mappings: list[EvidenceMapping]


class RewrittenBullet(StructuredOutput):
    text: str
    fact_ids: list[str]


class RewrittenRole(StructuredOutput):
    source_fact_id: str
    employer: str
    title: str
    functional_descriptor: str
    bullets: list[RewrittenBullet]


class RewrittenProject(StructuredOutput):
    source_fact_id: str
    bullets: list[RewrittenBullet]


class RewrittenSkills(StructuredOutput):
    category: str
    skills: list[str]


class RewriteOutput(StructuredOutput):
    summary: str
    experience: list[RewrittenRole]
    projects: list[RewrittenProject]
    skills: list[RewrittenSkills]
    content_decisions: list[str]
    clarification_questions: list[str]


def api_schema(model: type[BaseModel]) -> dict:
    """Remove constraints Claude doesn't support; Pydantic enforces them locally."""
    def transform(value):
        if isinstance(value, list):
            return [transform(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {
            key: ({name: transform(child) for name, child in item.items()}
                  if key in {"properties", "$defs", "definitions"} and isinstance(item, dict)
                  else transform(item))
            for key, item in value.items()
            if key not in {"minimum", "maximum", "minLength", "maxLength", "maxItems", "title"}
        }

    return transform(model.model_json_schema())
