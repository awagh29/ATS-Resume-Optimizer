"""Conservative, source-scoped checks for rewritten resume claims."""

from __future__ import annotations

import re

from app.models.schemas import ClaimAuditResult, OptimizedBullet, OptimizedResume, ResumeFacts, SupplementalFacts
from app.pipeline.scoring import term_present


METRIC_PATTERN = re.compile(
    r"(?:\$\s*\d[\d,.]*(?:\s*[kmb])?|\b\d+(?:[.,]\d+)*\+?\s*(?:%|x\b|"
    r"users?\b|customers?\b|clients?\b|requests?\b|transactions?\b|reports?\b|services?\b|"
    r"teams?\b|engineers?\b|years?\b|months?\b|milliseconds?\b|seconds?\b|minutes?\b|ms\b|hours?\b))",
    re.I,
)

# Source IDs are provenance, never proof of semantic fidelity. General paraphrases
# still need human review; explicit tool and numerical additions can be checked.
KNOWN_TOOLS = {
    "Python", "Java", "JavaScript", "TypeScript", "Go", "C++", "C#", "Rust", "React",
    "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI", "Spring", "AWS", "Azure",
    "GCP", "Google Cloud", "Docker", "Kubernetes", "Terraform", "PostgreSQL", "MySQL",
    "MongoDB", "Redis", "Kafka", "Spark", "Airflow", "SQL", "NoSQL", "GraphQL", "gRPC",
    "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy", "Tableau", "Power BI",
}


def audit_optimized_resume(
    optimized: OptimizedResume,
    facts: ResumeFacts,
    supplemental: SupplementalFacts | None = None,
) -> tuple[OptimizedResume, ClaimAuditResult]:
    cleaned = optimized.model_copy(deep=True)
    result = ClaimAuditResult()
    supp = "\n".join(supplemental.confirmed_lines()) if supplemental else ""
    supplemental_items = {
        f"supplemental-{i}": line for i, line in enumerate(supplemental.confirmed_lines(), 1)
    } if supplemental else {}
    allowed_skills = {s.lower() for sg in facts.skills for s in sg.skills}
    if supplemental:
        allowed_skills.update(s.lower() for s in supplemental.skills)

    def remove(note: str, question: str = "") -> None:
        result.passed = False
        result.removed_claims.append(note)
        if question:
            result.clarification_questions.append(question)

    def attributed_to(line: str, src) -> bool:
        # A fact ID is the most precise attribution, particularly for promotions
        # at the same employer. Unique employer/project names also work in the UI.
        if term_present(src.fact_id, line):
            return True
        if hasattr(src, "employer"):
            return bool(src.employer and term_present(src.employer, line)
                        and sum(role.employer.lower() == src.employer.lower() for role in facts.experience) == 1)
        return bool(src.name and term_present(src.name, line)
                    and sum(project.name.lower() == src.name.lower() for project in facts.projects) == 1)

    def audit_bullets(bullets, src, context: str):
        source_bullets = {b.fact_id: b.text for b in src.bullets}
        scoped_supplemental = {fid: text for fid, text in supplemental_items.items() if attributed_to(text, src)}
        allowed_ids = {src.fact_id, *source_bullets, *scoped_supplemental}
        kept = []
        for bullet in bullets:
            if any(fid not in allowed_ids for fid in bullet.fact_ids):
                remove(f"Removed bullet citing unknown or unrelated source facts: {bullet.text}")
                continue
            selected = list(source_bullets.values()) if not bullet.fact_ids or src.fact_id in bullet.fact_ids else [source_bullets[fid] for fid in bullet.fact_ids if fid in source_bullets]
            selected.extend(scoped_supplemental[fid] for fid in bullet.fact_ids if fid in scoped_supplemental)
            corpus = "\n".join([context, *selected]).lower()
            ok, note, question = _audit_bullet(bullet, corpus, allowed_skills)
            if ok:
                bullet.source_role_fact_id = src.fact_id
                bullet.fact_ids = bullet.fact_ids or [src.fact_id]
                kept.append(bullet)
            else:
                remove(note, question)
        return kept

    known_roles = {r.fact_id: r for r in facts.experience}
    kept_roles = []
    for role in cleaned.experience:
        src = known_roles.get(role.source_fact_id)
        if src is None:
            matches = [r for r in facts.experience if r.employer.lower() == role.employer.lower() and r.title.lower() == role.title.lower()]
            src = matches[0] if len(matches) == 1 else None
        if src is None:
            remove(f"Removed unknown employment role '{role.title}' at '{role.employer}'.")
            continue
        for field in ("employer", "title", "location", "start_date", "end_date"):
            if getattr(role, field) != getattr(src, field):
                remove(f"Restored official {field.replace('_', ' ')} '{getattr(src, field)}'.")
            setattr(role, field, getattr(src, field))
        role.source_fact_id = src.fact_id
        role.functional_descriptor = ""
        role.bullets = audit_bullets(role.bullets, src, f"{src.title} at {src.employer}")
        kept_roles.append(role)
    cleaned.experience = kept_roles

    known_projects = {p.fact_id: p for p in facts.projects}
    kept_projects = []
    for project in cleaned.projects:
        src = known_projects.get(project.source_fact_id)
        if src is None:
            matches = [p for p in facts.projects if p.name.lower() == project.name.lower()]
            src = matches[0] if len(matches) == 1 else None
        if src is None:
            remove(f"Removed unknown project '{project.name}'.")
            continue
        project.name, project.technologies, project.url = src.name, list(src.technologies), src.url
        project.source_fact_id = src.fact_id
        project.bullets = audit_bullets(project.bullets, src, f"{src.name}\n{src.description}\n{' '.join(src.technologies)}")
        kept_projects.append(project)
    cleaned.projects = kept_projects

    full_source = (facts.plain_text() + "\n" + supp).lower()
    for group in cleaned.skills:
        kept = []
        for skill in group.skills:
            if skill.lower() in allowed_skills or term_present(skill, full_source):
                kept.append(skill)
            else:
                remove(f"Removed unsupported skill '{skill}'.")
        group.skills = kept
    cleaned.skills = [sg for sg in cleaned.skills if sg.skills]
    if cleaned.summary:
        ok, note, question = _audit_bullet(OptimizedBullet(text=cleaned.summary), full_source, allowed_skills)
        if not ok:
            remove(note, question)
            cleaned.summary = ""
    cleaned.contact = facts.contact.model_copy(deep=True)
    cleaned.education = [e.model_copy(deep=True) for e in facts.education]
    cleaned.certifications = list(facts.certifications)
    result.notes.append(
        "Unsupported additions were removed; review paraphrases for factual accuracy."
        if result.removed_claims else
        "Source IDs, metrics and known tools passed source checks; review paraphrases for factual accuracy."
    )
    return cleaned, result


def _audit_bullet(bullet: OptimizedBullet, source_corpus: str, allowed_skills: set[str]) -> tuple[bool, str, str]:
    text = bullet.text.strip()
    if not text:
        return False, "Empty bullet removed.", ""
    for match in METRIC_PATTERN.finditer(text):
        metric = match.group(0)
        compact = re.sub(r"[,\s]", "", metric.lower())
        source_metrics = {re.sub(r"[,\s]", "", m.group(0).lower()) for m in METRIC_PATTERN.finditer(source_corpus)}
        if compact not in source_metrics:
            return False, f"Removed bullet with unsupported metric '{metric}': {text}", f"Can you confirm '{metric}' for this source experience?"
    for tool in KNOWN_TOOLS:
        if term_present(tool, text) and not term_present(tool, source_corpus):
            return False, f"Removed bullet with unsupported tool '{tool}': {text}", f"Did this source experience use {tool}?"
    stop = {"using", "with", "that", "from", "this", "built", "developed", "implemented", "created", "designed", "delivered", "resulting", "result", "improved", "across", "through", "based", "team", "system", "systems", "application", "applications", "service", "services"}
    significant = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9+.#/-]{3,}", text) if w.lower() not in stop]
    if significant and sum(term_present(w, source_corpus) for w in significant) / len(significant) < 0.25:
        return False, f"Removed weakly grounded bullet: {text}", f"What source experience supports: {text[:140]}?"
    return True, "", ""
