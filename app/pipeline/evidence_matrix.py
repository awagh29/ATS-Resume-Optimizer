"""Map job requirements to resume evidence without inventing facts."""

from __future__ import annotations

import re
from typing import Any

from app.models.schemas import (
    Classification,
    ExperienceBullet,
    ExperienceRole,
    EvidenceStrength,
    JobAnalysis,
    OptimizedResume,
    ProjectItem,
    RequirementEvidence,
    RequirementItem,
    ResumeFacts,
    SupplementalFacts,
)
from app.pipeline.scoring import STRENGTH_COVERAGE, TERM_ALIASES, term_present


def _supports_requirement(requirement: str, cited_text: str) -> bool:
    """A real quote must also contain substantive evidence for the requirement."""
    stop = {
        "and", "the", "with", "for", "you", "will", "have", "ability", "experience",
        "years", "using", "from", "that", "this", "work", "team", "strong", "knowledge",
        "including", "related", "field", "required", "preferred", "expertise", "proficiency",
        "proficient", "understanding", "familiarity", "demonstrated", "excellent", "proven",
        "build", "built", "building", "develop", "developed", "developing", "implement",
        "implemented", "design", "designed", "responsible", "software", "engineering",
    }
    tokens = {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{2,}", requirement)} - stop
    if any(term_present(token, cited_text) for token in tokens):
        return True
    # Preserve established multi-word technology aliases such as ML and k8s.
    if any(term_present(term, requirement) and term_present(term, cited_text) for term in TERM_ALIASES):
        return True
    equivalent_forms = (
        {"leadership", "lead", "led", "leading", "mentor", "mentored", "mentoring"},
        {"collaboration", "collaborated", "collaborate", "cross-functional"},
        {"testing", "tests", "test"},
        {"analysis", "analytics", "analyze", "analyzed", "analytical"},
        {"automation", "automated", "automate", "automating"},
        {"reliability", "reliable"},
        {"scale", "scaling", "scalability", "scaled"},
        {"deployment", "deployed", "deploy", "deploying"},
        {"monitoring", "monitor", "monitored", "observability"},
    )
    return any(tokens.intersection(forms) and any(term_present(form, cited_text) for form in forms) for forms in equivalent_forms)


def _fact_texts(facts: ResumeFacts, supp_lines: list[str] | None = None) -> dict[str, str]:
    index: dict[str, str] = {}
    for role in facts.experience:
        index[role.fact_id] = " ".join([role.employer, role.title, role.start_date, role.end_date, *(b.text for b in role.bullets)])
        index.update({b.fact_id: b.text for b in role.bullets})
    for project in facts.projects:
        index[project.fact_id] = " ".join([project.name, project.description, *project.technologies, *(b.text for b in project.bullets)])
        index.update({b.fact_id: b.text for b in project.bullets})
    index.update({sg.fact_id: ", ".join(sg.skills) for sg in facts.skills})
    index.update({edu.fact_id: " ".join([edu.institution, edu.degree, edu.field, *edu.details]) for edu in facts.education})
    index.update({f"supplemental-{i}": text for i, text in enumerate(supp_lines or [], 1)})
    return index


def _has_ci_evidence(text: str) -> bool:
    return term_present("GitHub Actions", text) and any(term_present(term, text) for term in ("CI", "testing", "checks", "pipeline"))


def build_evidence_matrix(
    analysis: JobAnalysis,
    facts: ResumeFacts,
    supplemental: SupplementalFacts | None = None,
    llm_mappings: list[dict[str, Any]] | None = None,
) -> list[RequirementEvidence]:
    resume_text = facts.plain_text()
    supp_lines = supplemental.confirmed_lines() if supplemental else []
    supp_text = "\n".join(supp_lines)
    corpus = resume_text + "\n" + supp_text

    llm_by_id = {}
    if llm_mappings:
        for m in llm_mappings:
            if isinstance(m, dict) and m.get("req_id"):
                llm_by_id[str(m["req_id"])] = m

    matrix: list[RequirementEvidence] = []
    for req in analysis.requirements:
        llm = llm_by_id.get(req.req_id, {})
        entry = _match_requirement(req, facts, corpus, supp_text, llm, supp_lines)
        matrix.append(entry)
    return matrix


def _match_requirement(
    req: RequirementItem,
    facts: ResumeFacts,
    corpus: str,
    supp_text: str,
    llm: dict[str, Any],
    supp_lines: list[str] | None = None,
) -> RequirementEvidence:
    # Start with deterministic keyword/overlap matching
    strength, evidence, fact_ids, coverage = _deterministic_match(req, facts, corpus, supp_text, supp_lines)

    # Allow LLM to upgrade only when it cites fact ids that exist
    if llm:
        llm_strength = str(llm.get("evidence_strength") or "").lower()
        raw_ids = llm.get("evidence_fact_ids") or []
        llm_fact_ids = [str(x) for x in raw_ids if x] if isinstance(raw_ids, list) else []
        index = _fact_texts(facts, supp_lines)
        valid_ids = set(index)
        cited = [fid for fid in llm_fact_ids if fid in valid_ids]
        if cited and llm_strength in {e.value for e in EvidenceStrength}:
            proposed = EvidenceStrength(llm_strength)
            # Never upgrade to DIRECT without cited facts and textual overlap
            if proposed == EvidenceStrength.DIRECT and not cited:
                proposed = EvidenceStrength.PARTIAL
            # Don't invent stronger than DIRECT if deterministic found nothing and LLM text empty
            llm_evidence = str(llm.get("matching_evidence") or "").strip()
            cited_text = "\n".join(index[fid] for fid in cited)
            normalized_evidence = re.sub(r"\s+", " ", llm_evidence.lower())
            normalized_source = re.sub(r"\s+", " ", cited_text.lower())
            if (llm_evidence and len(cited) == len(llm_fact_ids)
                    and normalized_evidence in normalized_source
                    and _supports_requirement(req.requirement, llm_evidence)):
                # Take the stronger of deterministic vs validated LLM
                order = [
                    EvidenceStrength.NONE,
                    EvidenceStrength.PARTIAL,
                    EvidenceStrength.TRANSFERABLE,
                    EvidenceStrength.DIRECT,
                ]
                if order.index(proposed) > order.index(strength):
                    strength = proposed
                    evidence = llm_evidence[:500]
                    fact_ids = cited
                    coverage = STRENGTH_COVERAGE[strength]

        treatment = str(llm.get("recommended_treatment") or "").strip()
    else:
        treatment = ""

    if not treatment:
        treatment = _default_treatment(req, strength)

    if _unverified_constraints(req.requirement, corpus) and coverage > 0.4:
        strength, coverage = EvidenceStrength.PARTIAL, 0.4
        treatment = "Related evidence exists, but this qualification needs explicit confirmation."

    return RequirementEvidence(
        req_id=req.req_id,
        requirement=req.requirement,
        excerpt=req.excerpt,
        classification=req.classification,
        importance=req.importance,
        rationale=req.rationale,
        matching_evidence=evidence,
        evidence_fact_ids=fact_ids,
        evidence_strength=strength,
        current_coverage=coverage,
        recommended_treatment=treatment,
    )


def _deterministic_match(
    req: RequirementItem,
    facts: ResumeFacts,
    corpus: str,
    supp_text: str,
    supp_lines: list[str] | None = None,
) -> tuple[EvidenceStrength, str, list[str], float]:
    req_low = req.requirement.lower()
    tokens = [
        t
        for t in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{2,}", req.requirement)
        if t.lower()
        not in {
            "and",
            "the",
            "with",
            "for",
            "you",
            "will",
            "have",
            "ability",
            "experience",
            "years",
            "using",
            "from",
            "that",
            "this",
            "work",
            "team",
            "strong",
            "knowledge",
            "including",
            "related",
            "field",
            "etc",
        }
    ]
    tokens = [token for token in tokens if token.lower() not in {
        "proficiency", "proficient", "expertise", "familiarity", "understanding", "excellent", "proven", "demonstrated",
    }]

    # Skills direct match
    skill_hits: list[str] = []
    skill_fact_ids: list[str] = []
    for sg in facts.skills:
        for skill in sg.skills:
            if term_present(skill, req.requirement) or term_present(skill, req_low):
                skill_hits.append(skill)
                skill_fact_ids.append(sg.fact_id)
            else:
                # reverse: requirement term in skills
                for tok in tokens:
                    if term_present(tok, skill):
                        skill_hits.append(skill)
                        skill_fact_ids.append(sg.fact_id)
                        break

    # Bullet matches
    best_bullet = ""
    best_ids: list[str] = []
    best_score = 0
    for role in facts.experience:
        for b in role.bullets:
            score = _overlap_score(tokens, b.text)
            # GitHub Actions plus explicit CI/test checks is real pipeline
            # evidence. It supports transferable CI/CD coverage without claiming
            # an unmentioned continuous deployment process.
            if (term_present("ci/cd", req.requirement) or term_present("continuous integration", req.requirement)) and _has_ci_evidence(b.text):
                score = max(score, 2)
            if score > best_score:
                best_score = score
                best_bullet = b.text
                best_ids = [b.fact_id, role.fact_id]
    for proj in facts.projects:
        for b in proj.bullets:
            score = _overlap_score(tokens, b.text)
            if score > best_score:
                best_score = score
                best_bullet = f"[Project: {proj.name}] {b.text}"
                best_ids = [b.fact_id, proj.fact_id]

    supp_hit = ""
    supp_id = ""
    if supp_text:
        for i, line in enumerate(supp_lines if supp_lines is not None else supp_text.splitlines(), 1):
            if _overlap_score(tokens, line) >= max(2, best_score):
                supp_hit = line
                supp_id = f"supplemental-{i}"
                break

    if skill_hits and best_score >= 2:
        return (
            EvidenceStrength.DIRECT,
            f"Skills: {', '.join(dict.fromkeys(skill_hits))}. Evidence: {best_bullet}",
            list(dict.fromkeys(skill_fact_ids + best_ids)),
            1.0,
        )
    if best_score >= 3 or (best_score >= 2 and skill_hits):
        return (EvidenceStrength.DIRECT, best_bullet, best_ids, 1.0)
    if skill_hits:
        return (
            EvidenceStrength.DIRECT,
            f"Listed skills: {', '.join(dict.fromkeys(skill_hits))}",
            list(dict.fromkeys(skill_fact_ids)),
            0.85,
        )
    if supp_hit:
        return (EvidenceStrength.DIRECT, supp_hit, [supp_id], 0.9)
    if best_score == 1 and len(tokens) == 1:
        return (EvidenceStrength.DIRECT, best_bullet, best_ids, 0.85)
    if best_score == 2:
        return (EvidenceStrength.TRANSFERABLE, best_bullet, best_ids, 0.7)
    if best_score == 1:
        return (EvidenceStrength.PARTIAL, best_bullet or "Weak lexical overlap only.", best_ids, 0.4)

    # Title/employer soft match for leadership etc.
    if re.search(r"\b(lead|mentor|manage|collaborate)\b", req_low):
        for role in facts.experience:
            if re.search(r"\b(lead|senior|manager|mentor)\b", f"{role.title} {role.employer}".lower()):
                return (
                    EvidenceStrength.PARTIAL,
                    f"Title/context: {role.title} at {role.employer}",
                    [role.fact_id],
                    0.4,
                )

    return (EvidenceStrength.NONE, "", [], 0.0)


def _overlap_score(tokens: list[str], text: str) -> int:
    if not text:
        return 0
    hits = 0
    seen = set()
    for tok in tokens:
        key = tok.lower()
        if key in seen:
            continue
        if term_present(tok, text):
            hits += 1
            seen.add(key)
    return hits


def _unverified_constraints(requirement: str, corpus: str) -> bool:
    """A matching tool cannot establish tenure, a degree, or work eligibility."""
    low = requirement.lower()
    from app.pipeline.analyze_jd import _extract_tech
    if any(not term_present(tool, corpus) and not (tool == "CI/CD" and _has_ci_evidence(corpus)) for tool in _extract_tech(requirement)):
        return True
    for match in re.finditer(r"\b\d+\+?\s*(?:years?|months?)\b", low):
        if match.group(0) not in corpus.lower():
            return True
    for pattern in (r"\b(?:bachelor|master|doctorate|phd|citizenship|citizen|clearance|authorized|authorization)\w*\b",):
        if any(not term_present(m.group(0), corpus) for m in re.finditer(pattern, low)):
            return True
    return False


def _default_treatment(req: RequirementItem, strength: EvidenceStrength) -> str:
    if strength == EvidenceStrength.DIRECT:
        return "Lead with this evidence; sharpen STAR outcome wording."
    if strength == EvidenceStrength.TRANSFERABLE:
        return "Reframe transferable evidence toward the employer's language without inventing tools."
    if strength == EvidenceStrength.PARTIAL:
        return "Keep as supporting detail; ask user for clarifying metrics or scope if available."
    if req.classification == Classification.REQUIRED:
        return "Do not invent coverage. Surface as mandatory gap with a concrete next step."
    if req.classification == Classification.PREFERRED:
        return "Omit unsupported claim; note as preferred gap if high importance."
    return "Treat as inference only; do not present as an employer requirement on the resume."


def rematch_matrix_for_text(
    matrix: list[RequirementEvidence],
    resume_text: str,
    *,
    facts: ResumeFacts | None = None,
    optimized: OptimizedResume | None = None,
    supplemental: SupplementalFacts | None = None,
) -> list[RequirementEvidence]:
    """Recompute coverage against a new resume text while preserving requirement metadata."""
    if facts is not None and optimized is not None:
        return _rematch_retained_facts(matrix, resume_text, facts, optimized, supplemental)
    updated: list[RequirementEvidence] = []
    for m in matrix:
        if m.evidence_strength == EvidenceStrength.NONE or m.current_coverage == 0:
            updated.append(m.model_copy(update={"evidence_strength": EvidenceStrength.NONE, "current_coverage": 0.0}))
            continue
        tokens = [
            t
            for t in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{2,}", m.requirement)
            if len(t) > 2 and t.lower() not in {"the", "and", "with", "for", "experience", "years", "ability", "strong", "knowledge", "required", "using", "have", "will"}
        ]
        score = _overlap_score(tokens, resume_text)
        if score >= 3:
            strength = EvidenceStrength.DIRECT
        elif score == 2:
            strength = EvidenceStrength.TRANSFERABLE
        elif score == 1:
            strength = EvidenceStrength.PARTIAL
        else:
            # keep prior if original evidence still appears
            if m.matching_evidence and m.matching_evidence[:40].lower() in resume_text.lower():
                strength = m.evidence_strength
            else:
                strength = EvidenceStrength.NONE
        coverage = STRENGTH_COVERAGE[strength]
        # Presentation can retain or lose supported evidence; keyword overlap can
        # never manufacture a qualification missing from the source matrix.
        if m.matching_evidence and m.matching_evidence.lower() in resume_text.lower():
            coverage, strength = m.current_coverage, m.evidence_strength
        coverage = min(coverage, m.current_coverage)
        if coverage == 0:
            strength = EvidenceStrength.NONE

        updated.append(
            m.model_copy(
                update={
                    "evidence_strength": strength,
                    "current_coverage": round(coverage, 2),
                }
            )
        )
    return updated


def _rematch_retained_facts(
    matrix: list[RequirementEvidence],
    final_text: str,
    facts: ResumeFacts,
    optimized: OptimizedResume,
    supplemental: SupplementalFacts | None,
) -> list[RequirementEvidence]:
    """Assess actual final content while recognizing provenance of paraphrases.

    A role ID is not proof that its original bullets survived layout fitting.
    Granular source IDs remain valid through a faithful audited paraphrase; a
    retained skill-group ID likewise does not mean all original skills survived.
    """
    source_bullets = {b.fact_id: b for entry in [*facts.experience, *facts.projects] for b in entry.bullets}
    source_entries = {entry.fact_id: entry for entry in [*facts.experience, *facts.projects]}
    source_skills = {group.fact_id: group for group in facts.skills}
    retained_ids = {fid for entry in [*optimized.experience, *optimized.projects] for bullet in entry.bullets for fid in bullet.fact_ids}
    valid_granular_ids = set(source_bullets)
    if supplemental:
        valid_granular_ids.update(f"supplemental-{i}" for i, _ in enumerate(supplemental.confirmed_lines(), 1))
    retained_ids &= valid_granular_ids
    retained_education = {edu.fact_id for edu in optimized.education}
    final_skill_text = ", ".join(skill for group in optimized.skills for skill in group.skills)

    # Build an independent view of what the final PDF actually says. This path
    # is used when source evidence was omitted, and can establish only coverage
    # already supported by the validated source matrix.
    final_facts = ResumeFacts(contact=optimized.contact, summary=optimized.summary,
                              education=optimized.education, skills=optimized.skills,
                              certifications=optimized.certifications)
    for entry in optimized.experience:
        final_facts.experience.append(ExperienceRole(
            fact_id=entry.source_fact_id, employer=entry.employer, title=entry.title,
            location=entry.location, start_date=entry.start_date, end_date=entry.end_date,
            bullets=[ExperienceBullet(fact_id=next((fid for fid in bullet.fact_ids if fid in source_bullets), entry.source_fact_id), text=bullet.text) for bullet in entry.bullets],
        ))
    for entry in optimized.projects:
        final_facts.projects.append(ProjectItem(
            fact_id=entry.source_fact_id, name=entry.name, technologies=entry.technologies,
            url=entry.url,
            bullets=[ExperienceBullet(fact_id=next((fid for fid in bullet.fact_ids if fid in source_bullets), entry.source_fact_id), text=bullet.text) for bullet in entry.bullets],
        ))

    def cited_evidence_retained(row: RequirementEvidence) -> bool:
        granular = set(row.evidence_fact_ids) & set(source_bullets)
        granular.update(fid for fid in row.evidence_fact_ids if fid.startswith("supplemental-"))
        # Claude sometimes cites a containing role/project. Resolve the quoted
        # evidence back to its actual bullet rather than retaining the whole role.
        if not granular:
            quote = re.sub(r"\s+", " ", row.matching_evidence.lower()).strip()
            for fid in row.evidence_fact_ids:
                entry = source_entries.get(fid)
                if entry is None:
                    continue
                for bullet in entry.bullets:
                    text = re.sub(r"\s+", " ", bullet.text.lower()).strip()
                    if quote and (quote in text or text in quote):
                        granular.add(bullet.fact_id)
        if granular:
            return granular <= retained_ids
        cited_education = set(row.evidence_fact_ids) & {edu.fact_id for edu in facts.education}
        if cited_education:
            return cited_education <= retained_education
        relevant_skills = {
            skill for fid in row.evidence_fact_ids if fid in source_skills
            for skill in source_skills[fid].skills
            if _supports_requirement(row.requirement, skill)
        }
        return bool(relevant_skills and all(term_present(skill, final_skill_text) for skill in relevant_skills))

    updated = []
    for row in matrix:
        if row.current_coverage == 0 or row.evidence_strength == EvidenceStrength.NONE:
            updated.append(row.model_copy(update={"current_coverage": 0.0, "evidence_strength": EvidenceStrength.NONE}))
            continue
        req = RequirementItem(req_id=row.req_id, requirement=row.requirement,
                              classification=row.classification, importance=row.importance,
                              excerpt=row.excerpt, rationale=row.rationale)
        strength, evidence, cited_ids, coverage = _deterministic_match(req, final_facts, final_text, "")
        if cited_evidence_retained(row):
            strength, coverage = row.evidence_strength, row.current_coverage
            evidence, cited_ids = row.matching_evidence, row.evidence_fact_ids
        coverage = min(coverage, row.current_coverage)
        if _unverified_constraints(row.requirement, final_text) and coverage > 0.4:
            strength, coverage = EvidenceStrength.PARTIAL, 0.4
        if coverage == 0:
            strength = EvidenceStrength.NONE
        updated.append(row.model_copy(update={
            "evidence_strength": strength, "current_coverage": round(coverage, 2),
            "matching_evidence": evidence if coverage else "",
            "evidence_fact_ids": list(dict.fromkeys(cited_ids)) if coverage else [],
        }))
    return updated
