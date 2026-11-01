"""Deterministic rewrite helpers and LLM result merging for optimized resumes."""

from __future__ import annotations

import re
from typing import Any

from app.models.schemas import (
    BulletChange,
    ContactInfo,
    EvidenceStrength,
    ExperienceBullet,
    GapItem,
    Classification,
    JobAnalysis,
    OptimizedBullet,
    OptimizedProject,
    OptimizedResume,
    OptimizedRole,
    PositioningChoice,
    RequirementEvidence,
    ResumeFacts,
    SkillGroup,
    SupplementalFacts,
)

ACTION_VERB_MAP = {
    "responsible for": "Owned",
    "helped with": "Supported",
    "worked on": "Contributed to",
    "was involved in": "Contributed to",
    "duties included": "Handled",
}


def choose_positioning(
    analysis: JobAnalysis,
    matrix: list[RequirementEvidence],
) -> PositioningChoice:
    """Pick strongest evidence-supported framing from JD priorities + evidence."""
    framings = {
        "technical_execution": 0.0,
        "product_ownership": 0.0,
        "operational_improvement": 0.0,
        "analytical_work": 0.0,
        "customer_impact": 0.0,
        "leadership": 0.0,
    }
    blob = " ".join(
        [
            analysis.role_title,
            " ".join(analysis.primary_responsibilities),
            " ".join(analysis.hiring_priorities),
            " ".join(analysis.repeated_themes),
        ]
    ).lower()

    if re.search(r"\b(build|implement|develop|engineer|code|architect)\b", blob):
        framings["technical_execution"] += 2
    if re.search(r"\b(product|roadmap|stakeholder|prioritiz)\b", blob):
        framings["product_ownership"] += 2
    if re.search(r"\b(ops|reliability|incident|automat|efficien|process)\b", blob):
        framings["operational_improvement"] += 2
    if re.search(r"\b(analy|data|metric|insight|experiment)\b", blob):
        framings["analytical_work"] += 2
    if re.search(r"\b(customer|user|client|support)\b", blob):
        framings["customer_impact"] += 2
    if re.search(r"\b(lead|mentor|manage|cross-functional)\b", blob):
        framings["leadership"] += 2

    for m in matrix:
        if m.evidence_strength in (EvidenceStrength.DIRECT, EvidenceStrength.TRANSFERABLE):
            req = m.requirement.lower()
            if re.search(r"\b(python|java|aws|kubernetes|api|system)\b", req):
                framings["technical_execution"] += m.current_coverage
            if re.search(r"\b(product|roadmap)\b", req):
                framings["product_ownership"] += m.current_coverage
            if re.search(r"\b(automat|pipeline|reliab|ops)\b", req):
                framings["operational_improvement"] += m.current_coverage
            if re.search(r"\b(analy|sql|data|metric)\b", req):
                framings["analytical_work"] += m.current_coverage
            if re.search(r"\b(customer|user)\b", req):
                framings["customer_impact"] += m.current_coverage
            if re.search(r"\b(lead|mentor|collabor)\b", req):
                framings["leadership"] += m.current_coverage

    ranked = sorted(framings.items(), key=lambda x: -x[1])
    best = ranked[0][0]
    labels = {
        "technical_execution": "Technical execution and delivery",
        "product_ownership": "Product ownership and prioritization",
        "operational_improvement": "Operational improvement and reliability",
        "analytical_work": "Analytical and data-informed work",
        "customer_impact": "Customer and user impact",
        "leadership": "Leadership and cross-functional collaboration",
    }
    alts = [labels[k] for k, _ in ranked[1:4]]
    return PositioningChoice(
        framing=labels[best],
        rationale=(
            f"Selected '{labels[best]}' because the job description emphasizes related themes "
            f"and the resume provides the strongest supporting evidence for this frame."
        ),
        alternatives_considered=alts,
    )


def deterministic_optimize(
    facts: ResumeFacts,
    analysis: JobAnalysis,
    matrix: list[RequirementEvidence],
    positioning: PositioningChoice,
    supplemental: SupplementalFacts | None = None,
) -> tuple[OptimizedResume, list[BulletChange], list[GapItem], list[dict[str, str]], list[str]]:
    """Build an optimized resume without inventing facts (used in demo mode and as fallback)."""
    changes: list[BulletChange] = []
    gaps: list[GapItem] = []
    terminology: list[dict[str, str]] = []
    decisions: list[str] = []

    # Relevance order for bullets: score by matrix keyword overlap
    important_terms = set()
    for m in matrix:
        if m.importance >= 0.5:
            for t in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{2,}", m.requirement):
                if len(t) > 2:
                    important_terms.add(t.lower())
    for t in analysis.tools_and_tech + analysis.domain_vocabulary:
        important_terms.add(t.lower())

    opt = OptimizedResume(contact=facts.contact.model_copy(), education=facts.education)

    # Summary only if positioning value
    if facts.summary or positioning.framing:
        summary_bits = []
        if facts.contact.name:
            # Don't start with name
            pass
        title_hint = analysis.role_title or "the target role"
        # Build from real skills + framing
        top_skills = []
        for sg in facts.skills:
            top_skills.extend(sg.skills)
        top_skills = top_skills[:6]
        evid = [m for m in matrix if m.evidence_strength == EvidenceStrength.DIRECT][:3]
        if top_skills or evid:
            skill_part = ", ".join(top_skills[:5]) if top_skills else "relevant technical experience"
            opt.summary = (
                f"{positioning.framing} professional targeting {title_hint}. "
                f"Strengths include {skill_part}."
            )
            # Ensure skills mentioned exist
            decisions.append("Added a short positioning summary grounded in listed skills.")
        else:
            opt.summary = ""

    # Experience: preserve order of roles but reorder/rewrite bullets
    for role in facts.experience:
        scored_bullets: list[tuple[float, ExperienceBullet, str]] = []
        for b in role.bullets:
            rewritten = _star_tighten(b.text, important_terms, analysis)
            score = _relevance(rewritten, important_terms)
            scored_bullets.append((score, b, rewritten))
            if rewritten != b.text:
                changes.append(
                    BulletChange(
                        before=b.text,
                        after=rewritten,
                        reason="Tightened to concise STAR-style accomplishment wording.",
                        fact_ids=[b.fact_id],
                    )
                )
        scored_bullets.sort(key=lambda x: -x[0])
        # Keep top bullets (max 5) preferring high relevance
        kept = scored_bullets[:5]
        if len(scored_bullets) > 5:
            decisions.append(
                f"Omitted {len(scored_bullets) - 5} lower-relevance bullets from {role.title} at {role.employer}."
            )
        opt_role = OptimizedRole(
            employer=role.employer,
            title=role.title,
            location=role.location,
            start_date=role.start_date,
            end_date=role.end_date,
            source_fact_id=role.fact_id,
            bullets=[
                OptimizedBullet(text=rw, fact_ids=[b.fact_id], source_role_fact_id=role.fact_id)
                for _, b, rw in kept
            ],
        )
        # Optional functional descriptor from positioning (short, not replacing title)
        if positioning.framing and "technical" in positioning.framing.lower():
            if not re.search(r"engineer|developer", role.title, re.I):
                opt_role.functional_descriptor = ""
        opt.experience.append(opt_role)

    # Projects: keep relevant ones
    for proj in facts.projects:
        bullets = []
        for b in proj.bullets:
            rw = _star_tighten(b.text, important_terms, analysis)
            if rw != b.text:
                changes.append(
                    BulletChange(
                        before=b.text,
                        after=rw,
                        reason="Project bullet tightened for outcome focus.",
                        fact_ids=[b.fact_id],
                    )
                )
            bullets.append(
                OptimizedBullet(text=rw, fact_ids=[b.fact_id], source_role_fact_id=proj.fact_id)
            )
        # relevance filter
        if bullets or any(t.lower() in important_terms for t in proj.technologies):
            opt.projects.append(
                OptimizedProject(
                    name=proj.name,
                    technologies=proj.technologies,
                    bullets=bullets[:4],
                    url=proj.url,
                    source_fact_id=proj.fact_id,
                )
            )
        else:
            decisions.append(f"Omitted lower-relevance project '{proj.name}'.")

    # Skills: reorder putting JD-relevant first, using only confirmed sources.
    skill_groups: list[SkillGroup] = []
    for sg in facts.skills:
        ordered = sorted(
            sg.skills,
            key=lambda s: (0 if s.lower() in important_terms or any(t in s.lower() for t in important_terms) else 1, s),
        )
        # Terminology added = JD terms that already exist in skills (aliases)
        for s in ordered:
            for tech in analysis.tools_and_tech:
                if s.lower() == tech.lower() or tech.lower() in s.lower():
                    terminology.append(
                        {
                            "term": tech,
                            "evidence": f"Already present in resume skills as '{s}'.",
                        }
                    )
        skill_groups.append(
            SkillGroup(fact_id=sg.fact_id, category=sg.category, skills=ordered)
        )
    opt.skills = skill_groups
    if supplemental:
        existing = {skill.casefold() for group in opt.skills for skill in group.skills}
        confirmed_skills = []
        for line in supplemental.skills:
            for skill in re.split(r"[,;|]", line):
                skill = skill.strip()
                if skill and skill.casefold() not in existing:
                    existing.add(skill.casefold())
                    confirmed_skills.append(skill)
        if confirmed_skills:
            confirmed_skills.sort(key=lambda skill: (-_relevance(skill, important_terms), skill.casefold()))
            taken_ids = {group.fact_id for group in opt.skills}
            group_id = "confirmed-skills"
            while group_id in taken_ids:
                group_id += "-additional"
            opt.skills.append(SkillGroup(fact_id=group_id, category="Additional skills", skills=confirmed_skills))
            decisions.append("Included additional skills explicitly confirmed in the knowledge base or supplemental facts.")
    opt.certifications = list(facts.certifications)

    if supplemental:
        _include_scoped_supplemental(opt, facts, supplemental, decisions)

    # Gaps from matrix
    for m in matrix:
        if m.evidence_strength == EvidenceStrength.NONE:
            gaps.append(
                GapItem(
                    requirement=m.requirement,
                    classification=m.classification,
                    gap_type="missing",
                    recommendation=(
                        "Do not claim this on the resume. "
                        + (
                            "Consider gaining this experience or certification."
                            if m.classification == Classification.REQUIRED
                            else "Optional to pursue if aligning to preferred qualifications."
                        )
                    ),
                    clarification_question="",
                )
            )
        elif m.evidence_strength == EvidenceStrength.PARTIAL:
            gaps.append(
                GapItem(
                    requirement=m.requirement,
                    classification=m.classification,
                    gap_type="needs_clarification",
                    recommendation="Strengthen with a verified metric or clearer personal contribution.",
                    clarification_question=f"What measurable outcome supports your experience with: {m.requirement[:120]}?",
                )
            )

    # Deduplicate terminology
    seen = set()
    uniq_term = []
    for t in terminology:
        key = t["term"].lower()
        if key not in seen:
            seen.add(key)
            uniq_term.append(t)

    return opt, changes[:12], gaps, uniq_term, decisions


def _include_scoped_supplemental(
    optimized: OptimizedResume,
    facts: ResumeFacts,
    supplemental: SupplementalFacts,
    decisions: list[str],
) -> None:
    """Attach confirmed lines only when the existing factual audit proves scope.

    Reusing the audit keeps demo and model-generated text under the same role-ID,
    unique-employer, unique-project and numerical provenance rules. Multiple
    possible targets remain unassigned rather than guessing employment history.
    """
    from app.pipeline.audit import audit_optimized_resume
    from app.pipeline.scoring import term_present

    attached = 0
    unassigned = 0
    entries = [*facts.experience, *facts.projects]
    for index, line in enumerate(supplemental.confirmed_lines(), 1):
        marker = re.match(r"^\[CONFIRMED (?:Metric|Responsibility|Accomplishment)\]\s*", line)
        if not marker:
            continue
        text = line[marker.end():].strip()
        fact_id = f"supplemental-{index}"
        explicit = [entry for entry in entries if term_present(entry.fact_id, text)]
        # A precise role/project ID takes precedence over a broad employer name.
        candidates = explicit if explicit else entries
        matches = []
        for source in candidates:
            label = source.employer if hasattr(source, "employer") else source.name
            alternatives = "|".join(re.escape(value) for value in (source.fact_id, label) if value)
            bullet_text = re.sub(
                rf"^(?:for|at|in|on)\s+(?:the\s+)?(?:project\s+)?(?:{alternatives})(?:\s+project)?\s*[:,]\s*",
                "", text, flags=re.I,
            ).strip()
            if not bullet_text:
                continue
            bullet = OptimizedBullet(text=bullet_text, fact_ids=[fact_id], source_role_fact_id=source.fact_id)
            if hasattr(source, "employer"):
                target = OptimizedRole(
                    source_fact_id=source.fact_id, employer=source.employer, title=source.title,
                    location=source.location, start_date=source.start_date, end_date=source.end_date,
                    bullets=[bullet],
                )
                candidate = OptimizedResume(experience=[target])
                checked, _ = audit_optimized_resume(candidate, facts, supplemental)
                valid = bool(checked.experience and checked.experience[0].bullets)
            else:
                target = OptimizedProject(
                    source_fact_id=source.fact_id, name=source.name,
                    technologies=list(source.technologies), url=source.url, bullets=[bullet],
                )
                candidate = OptimizedResume(projects=[target])
                checked, _ = audit_optimized_resume(candidate, facts, supplemental)
                valid = bool(checked.projects and checked.projects[0].bullets)
            if valid:
                matches.append(target)
        if len(matches) != 1:
            unassigned += 1
            continue
        accepted = matches[0]
        targets = optimized.experience if isinstance(accepted, OptimizedRole) else optimized.projects
        destination = next((entry for entry in targets if entry.source_fact_id == accepted.source_fact_id), None)
        if destination is None:
            targets.append(accepted)
            attached += 1
        elif not any(bullet.text.casefold().rstrip(".") == accepted.bullets[0].text.casefold().rstrip(".") for bullet in destination.bullets):
            destination.bullets.append(accepted.bullets[0])
            attached += 1
    if attached:
        decisions.append(f"Included {attached} confirmed supplemental facts with unambiguous source attribution.")
    if unassigned:
        decisions.append(f"Kept {unassigned} supplemental facts unassigned because their employer or project attribution needs clarification.")


def _relevance(text: str, terms: set[str]) -> float:
    low = text.lower()
    return float(sum(1 for t in terms if t in low))


def _star_tighten(text: str, important_terms: set[str], analysis: JobAnalysis) -> str:
    t = text.strip().rstrip(".")
    low = t.lower()
    for phrase, repl in ACTION_VERB_MAP.items():
        if low.startswith(phrase):
            t = repl + " " + t[len(phrase) :].lstrip()
            break
    # Capitalize first letter
    if t:
        t = t[0].upper() + t[1:]
    # Light terminology alignment only when equivalent skill already present in text
    for tech in analysis.tools_and_tech:
        aliases = {
            "k8s": "Kubernetes",
            "postgres": "PostgreSQL",
            "js": "JavaScript",
            "ts": "TypeScript",
            "gcp": "Google Cloud",
        }
        for alias, canonical in aliases.items():
            if alias in t.lower() and canonical.lower() in {x.lower() for x in analysis.tools_and_tech}:
                t = re.sub(rf"\b{alias}\b", canonical, t, flags=re.I)
    # Avoid adding "significantly" etc.
    t = re.sub(r"\b(significantly|various|several|multiple)\b\s*", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if not t.endswith("."):
        t += "."
    # Layout fitting may select whole bullets later. Never sever a factual sentence.
    return t


def merge_llm_optimized(
    fallback: OptimizedResume,
    llm_data: dict[str, Any],
    facts: ResumeFacts,
    supplemental: SupplementalFacts | None = None,
) -> OptimizedResume:
    """Merge LLM rewrite onto fallback while preserving employers/titles/dates from facts."""
    if not llm_data:
        return fallback

    opt = fallback.model_copy(deep=True)
    supplemental_ids = {
        f"supplemental-{i}" for i, _ in enumerate(supplemental.confirmed_lines(), 1)
    } if supplemental else set()
    if isinstance(llm_data.get("summary"), str) and llm_data["summary"].strip():
        opt.summary = llm_data["summary"].strip()[:500]

    llm_roles = llm_data.get("experience")
    if isinstance(llm_roles, list) and llm_roles:
        by_id = {r.fact_id: r for r in facts.experience}
        merged_roles: list[OptimizedRole] = []
        seen_roles: set[str] = set()
        for raw in llm_roles:
            if not isinstance(raw, dict):
                continue
            employer = str(raw.get("employer") or "").strip()
            title = str(raw.get("title") or "").strip()
            src = by_id.get(str(raw.get("source_fact_id") or ""))
            if not src:
                matches = [r for r in facts.experience if r.employer.lower() == employer.lower() and r.title.lower() == title.lower()]
                src = matches[0] if len(matches) == 1 else None
            if not src or src.fact_id in seen_roles:
                continue  # never invent employers
            seen_roles.add(src.fact_id)
            allowed_ids = {src.fact_id, *(b.fact_id for b in src.bullets), *supplemental_ids}
            bullets = []
            raw_bullets = raw.get("bullets")
            for b in raw_bullets if isinstance(raw_bullets, list) else []:
                if isinstance(b, dict):
                    text = str(b.get("text") or "").strip()
                    raw_ids = b.get("fact_ids") or []
                    if isinstance(raw_ids, list) and any(str(fid) not in allowed_ids for fid in raw_ids):
                        continue
                    fids = [str(x) for x in raw_ids if str(x) in allowed_ids] if isinstance(raw_ids, list) else []
                else:
                    text = str(b).strip()
                    fids = []
                if text:
                    bullets.append(
                        OptimizedBullet(
                            text=text,
                            fact_ids=fids or [src.fact_id],
                            source_role_fact_id=src.fact_id,
                        )
                    )
            if not bullets:
                # keep fallback bullets for this role
                fb = next((r for r in fallback.experience if r.source_fact_id == src.fact_id), None)
                bullets = fb.bullets if fb else []
            merged_roles.append(
                OptimizedRole(
                    employer=src.employer,
                    title=src.title,  # preserve official title
                    functional_descriptor="",
                    location=src.location,
                    start_date=src.start_date,
                    end_date=src.end_date,
                    bullets=bullets[:6],
                    source_fact_id=src.fact_id,
                )
            )
        if merged_roles:
            # A partial model response must not silently erase employment history.
            merged_roles.extend(r.model_copy(deep=True) for r in fallback.experience if r.source_fact_id not in seen_roles)
            opt.experience = merged_roles

    llm_projects = llm_data.get("projects")
    if isinstance(llm_projects, list):
        by_id = {p.fact_id: p for p in facts.projects}
        for raw in llm_projects:
            if not isinstance(raw, dict):
                continue
            src = by_id.get(str(raw.get("source_fact_id") or ""))
            if not src:
                continue
            target = next((p for p in opt.projects if p.source_fact_id == src.fact_id), None)
            if target is None:
                continue
            allowed_ids = {src.fact_id, *(b.fact_id for b in src.bullets), *supplemental_ids}
            bullets = []
            raw_bullets = raw.get("bullets")
            for b in raw_bullets if isinstance(raw_bullets, list) else []:
                if not isinstance(b, dict) or not isinstance(b.get("text"), str):
                    continue
                ids = b.get("fact_ids") or []
                if isinstance(ids, list) and any(str(fid) not in allowed_ids for fid in ids):
                    continue
                fids = [str(x) for x in ids if str(x) in allowed_ids] if isinstance(ids, list) else []
                if b["text"].strip():
                    bullets.append(OptimizedBullet(text=b["text"].strip(), fact_ids=fids or [src.fact_id], source_role_fact_id=src.fact_id))
            if bullets:
                target.bullets = bullets[:4]

    # Skills reorder from LLM only within existing skills
    llm_skills = llm_data.get("skills")
    if isinstance(llm_skills, list):
        allowed = {s.lower() for sg in facts.skills for s in sg.skills}
        if supplemental:
            allowed.update(skill.strip().lower() for line in supplemental.skills for skill in re.split(r"[,;|]", line) if skill.strip())
        new_groups: list[SkillGroup] = []
        for i, raw in enumerate(llm_skills):
            if not isinstance(raw, dict):
                continue
            raw_skills = raw.get("skills")
            skills = [
                str(s).strip()
                for s in (raw_skills if isinstance(raw_skills, list) else [])
                if str(s).strip().lower() in allowed
            ]
            if skills:
                new_groups.append(
                    SkillGroup(
                        fact_id=str(raw.get("fact_id") or f"sk{i+1}"),
                        category=str(raw.get("category") or "Skills")[:60],
                        skills=skills,
                    )
                )
        if new_groups:
            opt.skills = new_groups

    return opt


def build_recommendations(
    analysis: JobAnalysis,
    gaps: list[GapItem],
    changes: list[BulletChange],
    score_ceiling_reason: str,
    clarification_questions: list[str],
) -> tuple[list[str], list[str]]:
    recs: list[str] = []
    actions: list[str] = []

    if analysis.hiring_priorities:
        recs.append("Top hiring priorities: " + "; ".join(analysis.hiring_priorities[:3]))

    made = [c for c in changes[:3]]
    if made:
        recs.append(
            "Improvements already made: "
            + "; ".join(f"'{c.before[:40]}…' → '{c.after[:40]}…'" for c in made)
        )

    mandatory = [g for g in gaps if g.classification == Classification.REQUIRED]
    preferred = [g for g in gaps if g.classification == Classification.PREFERRED]
    if mandatory:
        recs.append(
            "Mandatory gaps (not claimed on resume): "
            + "; ".join(g.requirement[:80] for g in mandatory[:3])
        )
    if preferred:
        recs.append(
            "Preferred gaps: " + "; ".join(g.requirement[:80] for g in preferred[:2])
        )

    if score_ceiling_reason:
        recs.append(score_ceiling_reason)

    # Next actions (3–5)
    for g in mandatory[:2]:
        actions.append(
            f"Address mandatory gap — {g.requirement[:100]}: "
            + (g.recommendation or "Gain verifiable experience before claiming.")
        )
    for q in clarification_questions[:2]:
        actions.append(f"Confirm fact: {q}")
    if not actions:
        actions.append("Review the change log and confirm every rewritten bullet remains accurate.")
    actions.append("Re-run optimization after adding any confirmed metrics in Supplemental facts.")
    if len(actions) < 3:
        actions.append("Align LinkedIn summary with the selected positioning framing.")
    return recs[:8], actions[:5]
