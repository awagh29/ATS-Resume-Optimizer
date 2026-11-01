"""Pydantic schemas for structured pipeline data."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class EvidenceStrength(str, Enum):
    DIRECT = "direct"
    TRANSFERABLE = "transferable"
    PARTIAL = "partial"
    NONE = "none"


class Classification(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    INFERRED = "inferred"


class ContactInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""
    other_links: list[str] = Field(default_factory=list)


class ExperienceBullet(BaseModel):
    fact_id: str
    text: str
    source: str = "resume"  # resume | supplemental


class ExperienceRole(BaseModel):
    fact_id: str
    employer: str
    title: str
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    bullets: list[ExperienceBullet] = Field(default_factory=list)


class EducationItem(BaseModel):
    fact_id: str
    institution: str
    degree: str = ""
    field: str = ""
    location: str = ""
    graduation_date: str = ""
    details: list[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    fact_id: str
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    bullets: list[ExperienceBullet] = Field(default_factory=list)
    url: str = ""


class SkillGroup(BaseModel):
    fact_id: str
    category: str
    skills: list[str] = Field(default_factory=list)


class ResumeFacts(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = ""
    experience: list[ExperienceRole] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    skills: list[SkillGroup] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    other_sections: dict[str, list[str]] = Field(default_factory=dict)
    extraction_notes: list[str] = Field(default_factory=list)
    all_fact_ids: list[str] = Field(default_factory=list)

    def collect_fact_ids(self) -> list[str]:
        ids: list[str] = []
        for role in self.experience:
            ids.append(role.fact_id)
            ids.extend(b.fact_id for b in role.bullets)
        for edu in self.education:
            ids.append(edu.fact_id)
        for proj in self.projects:
            ids.append(proj.fact_id)
            ids.extend(b.fact_id for b in proj.bullets)
        for sg in self.skills:
            ids.append(sg.fact_id)
        self.all_fact_ids = ids
        return ids

    def plain_text(self) -> str:
        parts: list[str] = []
        c = self.contact
        parts.append(
            " | ".join(
                x
                for x in [c.name, c.email, c.phone, c.location, c.linkedin, c.github, c.website]
                if x
            )
        )
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.experience:
            parts.append("Experience")
        for role in self.experience:
            parts.append(
                f"{role.title} at {role.employer} ({role.start_date} – {role.end_date})"
            )
            for b in role.bullets:
                parts.append(f"- {b.text}")
        if self.education:
            parts.append("Education")
        for edu in self.education:
            parts.append(
                f"{edu.degree} {edu.field} — {edu.institution} ({edu.graduation_date})"
            )
        if self.projects:
            parts.append("Projects")
        for proj in self.projects:
            parts.append(f"Project: {proj.name}")
            if proj.description:
                parts.append(proj.description)
            for b in proj.bullets:
                parts.append(f"- {b.text}")
        if self.skills:
            parts.append("Skills")
        for sg in self.skills:
            parts.append(f"{sg.category}: {', '.join(sg.skills)}")
        if self.certifications:
            parts.append("Certifications: " + ", ".join(self.certifications))
        return "\n".join(parts)


class SupplementalFacts(BaseModel):
    metrics: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    accomplishments: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    raw_text: str = ""

    def confirmed_lines(self) -> list[str]:
        lines: list[str] = []
        for label, items in [
            ("Metric", self.metrics),
            ("Responsibility", self.responsibilities),
            ("Accomplishment", self.accomplishments),
            ("Skill", self.skills),
            ("Project", self.projects),
        ]:
            for item in items:
                item = item.strip()
                if item:
                    lines.append(f"[CONFIRMED {label}] {item}")
        return lines


class RequirementItem(BaseModel):
    req_id: str
    requirement: str
    excerpt: str = ""
    classification: Classification = Classification.REQUIRED
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    is_core: bool = True


class JobAnalysis(BaseModel):
    role_title: str = ""
    seniority: str = ""
    company_hint: str = ""
    primary_responsibilities: list[str] = Field(default_factory=list)
    required_qualifications: list[str] = Field(default_factory=list)
    preferred_qualifications: list[str] = Field(default_factory=list)
    tools_and_tech: list[str] = Field(default_factory=list)
    domain_vocabulary: list[str] = Field(default_factory=list)
    deliverables_and_outcomes: list[str] = Field(default_factory=list)
    leadership_collaboration: list[str] = Field(default_factory=list)
    eligibility_constraints: list[str] = Field(default_factory=list)
    repeated_themes: list[str] = Field(default_factory=list)
    hiring_priorities: list[str] = Field(default_factory=list)
    inferred_notes: list[str] = Field(default_factory=list)
    requirements: list[RequirementItem] = Field(default_factory=list)


class RequirementEvidence(BaseModel):
    req_id: str
    requirement: str
    excerpt: str = ""
    classification: Classification = Classification.REQUIRED
    importance: float = 0.5
    rationale: str = ""
    matching_evidence: str = ""
    evidence_fact_ids: list[str] = Field(default_factory=list)
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    current_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_treatment: str = ""


class PositioningChoice(BaseModel):
    framing: str
    rationale: str
    alternatives_considered: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    required_qualification_evidence: float = Field(default=0.0, ge=0.0, le=35.0)
    primary_responsibility_coverage: float = Field(default=0.0, ge=0.0, le=25.0)
    skills_and_terminology: float = Field(default=0.0, ge=0.0, le=15.0)
    evidence_and_accomplishment_quality: float = Field(default=0.0, ge=0.0, le=15.0)
    ats_readability_and_structure: float = Field(default=0.0, ge=0.0, le=10.0)
    total: float = Field(default=0.0, ge=0.0, le=100.0)
    mandatory_gaps: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def recompute_total(self) -> float:
        self.total = round(
            self.required_qualification_evidence
            + self.primary_responsibility_coverage
            + self.skills_and_terminology
            + self.evidence_and_accomplishment_quality
            + self.ats_readability_and_structure,
            1,
        )
        return self.total


class MatchAssessment(BaseModel):
    label: str = "Internal match estimate"
    score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    matrix: list[RequirementEvidence] = Field(default_factory=list)


class GapItem(BaseModel):
    requirement: str
    classification: Classification = Classification.REQUIRED
    gap_type: str = "missing"  # missing | partial | needs_clarification | needs_experience
    recommendation: str = ""
    clarification_question: str = ""


class BulletChange(BaseModel):
    before: str
    after: str
    reason: str
    fact_ids: list[str] = Field(default_factory=list)


class OptimizedBullet(BaseModel):
    text: str
    fact_ids: list[str] = Field(default_factory=list)
    source_role_fact_id: str = ""


class OptimizedRole(BaseModel):
    employer: str
    title: str
    functional_descriptor: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    bullets: list[OptimizedBullet] = Field(default_factory=list)
    source_fact_id: str = ""


class OptimizedProject(BaseModel):
    name: str
    technologies: list[str] = Field(default_factory=list)
    bullets: list[OptimizedBullet] = Field(default_factory=list)
    url: str = ""
    source_fact_id: str = ""


class OptimizedResume(BaseModel):
    contact: ContactInfo = Field(default_factory=ContactInfo)
    summary: str = ""
    experience: list[OptimizedRole] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    projects: list[OptimizedProject] = Field(default_factory=list)
    skills: list[SkillGroup] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class ClaimAuditResult(BaseModel):
    passed: bool = True
    removed_claims: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CompilationResult(BaseModel):
    success: bool = False
    pdf_bytes: Optional[bytes] = None
    page_count: Optional[int] = None
    extracted_text: str = ""
    error: str = ""
    compiler_available: bool = False
    layout_verified: bool = False
    warnings: list[str] = Field(default_factory=list)


class OptimizationResult(BaseModel):
    run_id: str = ""
    page_size: str = "letter"
    knowledge_used: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
    demo_mode: bool = False
    demo_notice: str = ""
    job_analysis: JobAnalysis = Field(default_factory=JobAnalysis)
    positioning: PositioningChoice = Field(
        default_factory=lambda: PositioningChoice(framing="", rationale="")
    )
    original_assessment: MatchAssessment = Field(default_factory=MatchAssessment)
    optimized_assessment: MatchAssessment = Field(default_factory=MatchAssessment)
    optimized_resume: OptimizedResume = Field(default_factory=OptimizedResume)
    latex_source: str = ""
    compilation: CompilationResult = Field(default_factory=CompilationResult)
    change_log: list[BulletChange] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    terminology_added: list[dict[str, str]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    content_decisions: list[str] = Field(default_factory=list)
    passes_used: int = 0
    target_reached: bool = False
    score_ceiling_reason: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def to_template_dict(self) -> dict[str, Any]:
        """JSON-serializable view for templates (excludes pdf bytes)."""
        data = self.model_dump(mode="json", exclude={"compilation": {"pdf_bytes"}})
        data["compilation"]["has_pdf"] = bool(self.compilation.pdf_bytes)
        data["compilation"]["pdf_bytes"] = None
        return data


def parse_supplemental_text(raw: str) -> SupplementalFacts:
    """Parse freeform supplemental text into categorized confirmed facts."""
    facts = SupplementalFacts(raw_text=raw.strip())
    if not raw.strip():
        return facts
    # Keep raw as confirmed source; also split non-empty lines as accomplishments
    # when user didn't use categories — still treated as confirmed.
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("metric:") or lower.startswith("metrics:"):
            facts.metrics.append(line.split(":", 1)[1].strip())
        elif lower.startswith("skill:") or lower.startswith("skills:"):
            facts.skills.append(line.split(":", 1)[1].strip())
        elif lower.startswith("project:") or lower.startswith("projects:"):
            facts.projects.append(line.split(":", 1)[1].strip())
        elif lower.startswith("responsibility:") or lower.startswith("responsibilities:"):
            facts.responsibilities.append(line.split(":", 1)[1].strip())
        elif lower.startswith("accomplishment:") or lower.startswith("accomplishments:"):
            facts.accomplishments.append(line.split(":", 1)[1].strip())
        else:
            facts.accomplishments.append(line)
    return facts


class OptimizeRequest(BaseModel):
    resume_latex: str
    job_description: str
    company: str = ""
    role_title: str = ""
    supplemental: str = ""
    page_size: str = "letter"

    @field_validator("page_size")
    @classmethod
    def validate_page_size(cls, v: str) -> str:
        v = (v or "letter").lower().strip()
        if v not in {"letter", "a4"}:
            raise ValueError("page_size must be 'letter' or 'a4'")
        return v
