"""
Internal match estimate scoring (0–100).

Rubric (documented, application-calculated):
  - Required qualification evidence:        35 pts
  - Coverage of primary responsibilities:   25 pts
  - Relevant skills and terminology:        15 pts
  - Evidence and accomplishment quality:    15 pts
  - ATS readability and document structure: 10 pts

Rules:
  - Same matrix/weights for original and optimized assessments.
  - Partial credit for partial evidence; no reward for keyword stuffing.
  - Preferred qualifications cannot mask missing mandatory ones.
  - Mandatory gaps are listed even when total score is high.
"""

from __future__ import annotations

import re
from typing import Iterable

from app.models.schemas import (
    Classification,
    EvidenceStrength,
    MatchAssessment,
    OptimizedResume,
    RequirementEvidence,
    ResumeFacts,
    ScoreBreakdown,
)

STRENGTH_COVERAGE = {
    EvidenceStrength.DIRECT: 1.0,
    EvidenceStrength.TRANSFERABLE: 0.7,
    EvidenceStrength.PARTIAL: 0.4,
    EvidenceStrength.NONE: 0.0,
}

# Common technology/term equivalents (not interchangeable domains — only aliases)
TERM_ALIASES: dict[str, set[str]] = {
    "javascript": {"js", "javascript", "ecmascript"},
    "typescript": {"ts", "typescript"},
    "python": {"python", "py"},
    "kubernetes": {"kubernetes", "k8s"},
    "postgresql": {"postgresql", "postgres", "psql"},
    "amazon web services": {"aws", "amazon web services"},
    "google cloud": {"gcp", "google cloud", "google cloud platform"},
    "machine learning": {"machine learning", "ml"},
    "artificial intelligence": {"artificial intelligence", "ai"},
    "continuous integration": {"ci", "continuous integration"},
    "continuous delivery": {"cd", "continuous delivery", "continuous deployment"},
    "ci/cd": {"ci/cd", "continuous integration and delivery", "continuous integration and continuous delivery", "continuous integration/continuous delivery"},
    "mentorship": {"mentorship", "mentor", "mentored", "mentoring"},
    "collaboration": {"collaboration", "collaborate", "collaborated", "collaborating"},
    "testing": {"testing", "tests", "test"},
    "rest": {"rest", "restful", "rest api"},
    "sql": {"sql", "structured query language"},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def expand_terms(term: str) -> set[str]:
    t = _normalize(term)
    out = {t}
    for canonical, aliases in TERM_ALIASES.items():
        if t == canonical or t in aliases:
            out |= aliases | {canonical}
    # also split multi-word lightly
    if " " in t:
        out.add(t)
    return out


def term_present(term: str, corpus: str) -> bool:
    corpus_n = _normalize(corpus)
    for alias in expand_terms(term):
        if not alias:
            continue
        # word-boundary-ish match
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, corpus_n):
            return True
    return False


def _unique_terms(terms: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = _normalize(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def score_from_matrix(
    matrix: list[RequirementEvidence],
    resume_text: str,
    tools_and_tech: list[str] | None = None,
    domain_vocabulary: list[str] | None = None,
    *,
    structure_score: float | None = None,
    quality_score: float | None = None,
) -> ScoreBreakdown:
    """
    Compute ScoreBreakdown from a requirement-evidence matrix and resume text.

    structure_score / quality_score may be supplied (0–1) by the caller;
    otherwise they are estimated heuristically from text.
    """
    breakdown = ScoreBreakdown()
    tools_and_tech = tools_and_tech or []
    domain_vocabulary = domain_vocabulary or []

    required = [m for m in matrix if m.classification == Classification.REQUIRED]
    preferred = [m for m in matrix if m.classification == Classification.PREFERRED]
    responsibilities = [
        m
        for m in matrix
        if "responsib" in _normalize(m.rationale)
        or m.req_id.startswith("resp")
        or "responsibility" in _normalize(m.requirement)
    ]
    # If no explicit responsibility rows, use high-importance required items as proxy
    if not responsibilities:
        responsibilities = [m for m in required if m.importance >= 0.6][:8]

    # --- Required qualification evidence (35) ---
    if required:
        weighted = 0.0
        weight_sum = 0.0
        for m in required:
            cov = m.current_coverage
            if cov <= 0:
                cov = STRENGTH_COVERAGE.get(m.evidence_strength, 0.0)
            w = max(m.importance, 0.2)
            weighted += cov * w
            weight_sum += w
            if cov < 0.5:
                breakdown.mandatory_gaps.append(m.requirement)
        ratio = weighted / weight_sum if weight_sum else 0.0
        breakdown.required_qualification_evidence = round(35.0 * ratio, 1)
    else:
        breakdown.required_qualification_evidence = 20.0
        breakdown.notes.append("No required qualifications extracted; partial default applied.")

    # Preferred cannot inflate beyond filling remaining headroom of required category —
    # they are intentionally NOT added to the 35-pt bucket.

    # --- Primary responsibility coverage (25) ---
    if responsibilities:
        scores = []
        for m in responsibilities:
            cov = m.current_coverage
            if cov <= 0:
                cov = STRENGTH_COVERAGE.get(m.evidence_strength, 0.0)
            scores.append(cov * max(m.importance, 0.3))
        avg = sum(scores) / len(scores)
        # normalize roughly (importance avg ~0.5–1)
        breakdown.primary_responsibility_coverage = round(min(25.0, 25.0 * avg / 0.85), 1)
    else:
        # fall back to overall matrix coverage
        if matrix:
            avg = sum(
                (m.current_coverage or STRENGTH_COVERAGE.get(m.evidence_strength, 0.0))
                for m in matrix
            ) / len(matrix)
            breakdown.primary_responsibility_coverage = round(25.0 * avg * 0.8, 1)
        else:
            breakdown.primary_responsibility_coverage = 10.0

    # --- Skills and terminology (15) — unique hits only ---
    terms = _unique_terms(list(tools_and_tech) + list(domain_vocabulary))
    if terms:
        hits = sum(1 for t in terms if term_present(t, resume_text))
        # diminishing returns after ~70% unique coverage
        ratio = hits / len(terms)
        breakdown.skills_and_terminology = round(15.0 * min(1.0, ratio / 0.7), 1)
    else:
        # use preferred/tech-like matrix rows
        tech_rows = [m for m in preferred + required if m.importance >= 0.4]
        if tech_rows:
            hits = sum(
                1
                for m in tech_rows
                if (m.current_coverage or STRENGTH_COVERAGE.get(m.evidence_strength, 0.0)) >= 0.4
            )
            breakdown.skills_and_terminology = round(15.0 * hits / len(tech_rows), 1)
        else:
            breakdown.skills_and_terminology = 7.0

    # --- Evidence and accomplishment quality (15) ---
    if quality_score is None:
        quality_score = estimate_accomplishment_quality(resume_text)
    breakdown.evidence_and_accomplishment_quality = round(15.0 * max(0.0, min(1.0, quality_score)), 1)

    # --- ATS readability and structure (10) ---
    if structure_score is None:
        structure_score = estimate_structure_score(resume_text)
    breakdown.ats_readability_and_structure = round(10.0 * max(0.0, min(1.0, structure_score)), 1)

    # Soft note if preferred gaps exist but score is high
    for m in preferred:
        cov = m.current_coverage or STRENGTH_COVERAGE.get(m.evidence_strength, 0.0)
        if cov < 0.4 and m.importance >= 0.7:
            breakdown.notes.append(f"Preferred gap: {m.requirement}")

    # Cap: if any critical required gap (importance>=0.85, coverage<0.3), soft-penalize total via note
    critical_gaps = [
        m
        for m in required
        if m.importance >= 0.85
        and (m.current_coverage or STRENGTH_COVERAGE.get(m.evidence_strength, 0.0)) < 0.3
    ]
    if critical_gaps:
        # Ensure mandatory gaps listed; slight reduction so preferred can't hide them
        penalty = min(8.0, 3.0 * len(critical_gaps))
        breakdown.required_qualification_evidence = max(
            0.0, breakdown.required_qualification_evidence - penalty
        )
        breakdown.notes.append(
            "Critical mandatory gaps reduced the required-qualification component."
        )

    breakdown.recompute_total()
    return breakdown


def estimate_accomplishment_quality(text: str) -> float:
    """Heuristic 0–1 quality from action verbs, specificity, metrics presence."""
    if not text.strip():
        return 0.2
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith(("-", "•", "*"))]
    if not lines:
        # treat sentences as pseudo-bullets
        lines = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 40][:20]
    if not lines:
        return 0.3

    action_verbs = {
        "built",
        "designed",
        "implemented",
        "led",
        "developed",
        "created",
        "optimized",
        "reduced",
        "increased",
        "delivered",
        "automated",
        "migrated",
        "launched",
        "improved",
        "analyzed",
        "architected",
        "deployed",
        "collaborated",
        "owned",
        "shipped",
        "refactored",
        "integrated",
        "established",
        "streamlined",
    }
    vague = {"responsible for", "helped with", "worked on", "various", "several", "significantly"}
    scored = 0.0
    for ln in lines:
        low = ln.lower()
        pts = 0.35
        first = re.sub(r"^[-•*\s]+", "", low).split(" ", 1)[0]
        if first in action_verbs:
            pts += 0.25
        if re.search(r"\d", ln):
            pts += 0.2
        if any(v in low for v in vague):
            pts -= 0.2
        if 40 <= len(ln) <= 220:
            pts += 0.1
        scored += max(0.0, min(1.0, pts))
    return max(0.15, min(1.0, scored / len(lines)))


def estimate_structure_score(text: str) -> float:
    """Heuristic ATS structure score 0–1."""
    if not text.strip():
        return 0.2
    low = text.lower()
    score = 0.3
    for heading in ("experience", "education", "skills", "projects"):
        if heading in low:
            score += 0.12
    if re.search(r"[\w.+-]+@[\w.-]+\.\w+", text):
        score += 0.1
    if re.search(r"\b(20\d{2}|19\d{2})\b", text):
        score += 0.08
    # penalize tables/icons language hints
    if any(x in low for x in ("tikz", "fontawesome", "includegraphics", "skill bar")):
        score -= 0.15
    return max(0.1, min(1.0, score))


def build_assessment(
    matrix: list[RequirementEvidence],
    resume_text: str,
    tools_and_tech: list[str] | None = None,
    domain_vocabulary: list[str] | None = None,
    structure_score: float | None = None,
    quality_score: float | None = None,
) -> MatchAssessment:
    score = score_from_matrix(
        matrix,
        resume_text,
        tools_and_tech=tools_and_tech,
        domain_vocabulary=domain_vocabulary,
        structure_score=structure_score,
        quality_score=quality_score,
    )
    return MatchAssessment(score=score, matrix=matrix)


def facts_to_text(facts: ResumeFacts) -> str:
    return facts.plain_text()


def optimized_to_text(opt: OptimizedResume) -> str:
    parts: list[str] = []
    c = opt.contact
    parts.append(
        " | ".join(
            x
            for x in [c.name, c.email, c.phone, c.location, c.linkedin, c.github, c.website]
            if x
        )
    )
    if opt.summary:
        parts.append("Summary")
        parts.append(opt.summary)
    if opt.experience:
        parts.append("Experience")
    for role in opt.experience:
        title = role.title
        if role.functional_descriptor:
            title = f"{role.title} — {role.functional_descriptor}"
        parts.append(f"{title} at {role.employer} ({role.start_date} – {role.end_date})")
        for b in role.bullets:
            parts.append(f"- {b.text}")
    if opt.education:
        parts.append("Education")
    for edu in opt.education:
        parts.append(f"{edu.degree} {edu.field} — {edu.institution}")
    if opt.projects:
        parts.append("Projects")
    for proj in opt.projects:
        parts.append(f"Project: {proj.name}")
        for b in proj.bullets:
            parts.append(f"- {b.text}")
    if opt.skills:
        parts.append("Skills")
    for sg in opt.skills:
        parts.append(f"{sg.category}: {', '.join(sg.skills)}")
    if opt.certifications:
        parts.append("Certifications: " + ", ".join(opt.certifications))
    return "\n".join(parts)
