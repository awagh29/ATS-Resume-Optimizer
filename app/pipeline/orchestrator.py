"""Evidence-backed Claude optimization, followed by strict one-page compilation."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3

from app.config import config_value
from app.latex.latex_render import fit_resume_to_one_page
from app.llm import prompts
from app.llm.adapter import LLMClient, LLMError, get_llm_client
from app.llm.demo import DEMO_NOTICE
from app.llm.outputs import PlanOutput, RewriteOutput
from app.llm.session import OptimizationSession
from app.models.schemas import (
    BulletChange, EvidenceStrength, GapItem, OptimizationResult, parse_supplemental_text,
)
from app.pipeline.analyze_jd import analyze_job_description, merge_llm_job_analysis
from app.pipeline.audit import audit_optimized_resume
from app.pipeline.evidence_matrix import build_evidence_matrix, rematch_matrix_for_text
from app.pipeline.parse_resume import parse_resume_latex
from app.pipeline.rewrite import (
    build_recommendations, choose_positioning, deterministic_optimize, merge_llm_optimized,
)
from app.pipeline.scoring import (
    build_assessment, estimate_accomplishment_quality, estimate_structure_score,
    facts_to_text, optimized_to_text, term_present,
)


def run_optimization(
    resume_latex: str,
    job_description: str,
    *,
    company: str = "",
    role_title: str = "",
    supplemental_text: str = "",
    page_size: str = "letter",
    llm: LLMClient | None = None,
    knowledge_store=None,
    use_knowledge: bool = True,
    profile_id: str = "default",
) -> OptimizationResult:
    if page_size not in {"letter", "a4"}:
        raise ValueError("Page size must be letter or a4.")
    for value, setting in (
        (resume_latex, "MAX_RESUME_CHARS"),
        (job_description, "MAX_JD_CHARS"),
        (supplemental_text, "MAX_SUPPLEMENTAL_CHARS"),
    ):
        if len(value) > config_value(setting):
            raise ValueError("Input exceeds the configured size limit.")
    if not resume_latex.strip() or not job_description.strip():
        raise ValueError("Resume and job description are required.")

    client = llm if llm is not None else get_llm_client()
    result = OptimizationResult(demo_mode=not client.configured, page_size=page_size)
    model_session = OptimizationSession(client, knowledge_store, profile_id)
    if result.demo_mode:
        result.demo_notice = DEMO_NOTICE
        result.warnings.append(DEMO_NOTICE)

    # Parse source identities before deriving scoped facts from stored notes.
    facts = parse_resume_latex(resume_latex)
    result.warnings.extend(facts.extraction_notes)
    if not (facts.experience or facts.projects or facts.education or facts.skills):
        raise ValueError("No resume sections could be extracted. Use standard LaTeX sections or the sample format.")
    supplemental = parse_supplemental_text(supplemental_text)
    knowledge = []
    if knowledge_store is not None and use_knowledge:
        try:
            knowledge = knowledge_store.retrieve(
                job_description, profile_id=profile_id,
                max_chars=config_value("KNOWLEDGE_MAX_CONTEXT_CHARS"),
                limit=config_value("KNOWLEDGE_MAX_ENTRIES_PER_RUN"),
            )
        except (sqlite3.Error, OSError, ValueError):
            result.warnings.append("Stored knowledge could not be loaded. This run uses the supplied resume and facts.")
    # Track each appended item's origin by category/index. Identical text can be
    # entered in the current form and multiple notes without merging provenance.
    knowledge_origins = {}
    category_fields = {"skill": "skills", "skills": "skills", "metric": "metrics",
                       "metrics": "metrics", "responsibility": "responsibilities",
                       "responsibilities": "responsibilities", "project": "projects",
                       "projects": "projects"}
    confirmed_fields = [("metrics", "Metric"), ("responsibilities", "Responsibility"),
                        ("accomplishments", "Accomplishment"), ("skills", "Skill"),
                        ("projects", "Project")]

    def supplemental_records():
        records = []
        for field, label in confirmed_fields:
            for index, text in enumerate(getattr(supplemental, field)):
                if not text.strip():
                    continue
                record = {"fact_id": f"supplemental-{len(records) + 1}",
                          "text": f"[CONFIRMED {label}] {text.strip()}"}
                origin = knowledge_origins.get((field, index))
                if origin is not None:
                    record["knowledge_id"] = origin["id"]
                records.append(record)
        return records

    scope_aliases = list(dict.fromkeys(
        alias for entry in [*facts.experience, *facts.projects]
        for alias in (entry.fact_id, getattr(entry, "employer", getattr(entry, "name", "")))
        if alias
    ))
    scope_headers = [re.compile(
        r"^(?:(?:at|for|employer|company|project|role)\s*:?\s*)?"
        + re.escape(alias) + r"\s*:?$", re.I,
    ) for alias in scope_aliases]
    skipped_knowledge = 0
    for entry in knowledge:
        field = category_fields.get(entry["category"], "accomplishments")
        scope = ""
        for line in entry["content"].splitlines():
            line = line.strip()
            if not line:
                continue
            if field != "skills" and any(pattern.fullmatch(line) for pattern in scope_headers):
                # A standalone known employer/project header scopes subsequent
                # lines; it is not itself an accomplishment.
                scope = line.rstrip(":").strip()
                continue
            mentions_source = any(term_present(alias, line) for alias in scope_aliases)
            # Explicit attribution on this line overrides any carried header.
            text = f"{scope}: {line}" if scope and not mentions_source else line
            items = getattr(supplemental, field)
            index = len(items)
            items.append(text)
            knowledge_origins[(field, index)] = entry
            derived = [item for item in supplemental_records() if "knowledge_id" in item]
            # Recompute final IDs in confirmation order and include their JSON
            # metadata/labels in the budget, not merely the original note text.
            if len(json.dumps(derived, ensure_ascii=False, separators=(",", ":"))) > config_value("KNOWLEDGE_MAX_CONTEXT_CHARS"):
                items.pop()
                knowledge_origins.pop((field, index))
                skipped_knowledge += 1
    if skipped_knowledge:
        result.warnings.append(
            "Some stored knowledge facts were skipped to keep the model context within its configured budget. "
            "Complete facts were retained without truncation."
        )
    # Preserve the full structured source, including exact identifiers and metadata.
    fact_payload = facts.model_dump(mode="json", exclude={"extraction_notes", "all_fact_ids"})
    supplemental_payload = supplemental_records()
    for entry in knowledge:
        fact_ids = [item["fact_id"] for item in supplemental_payload
                    if item.get("knowledge_id") == entry["id"]]
        if fact_ids:
            result.knowledge_used.append({"id": entry["id"], "title": entry["title"],
                                          "category": entry["category"], "content": entry["content"],
                                          "fact_ids": fact_ids})
    analysis = analyze_job_description(job_description, company=company, role_title=role_title)
    mappings = None
    if not result.demo_mode:
        response = model_session.request(
            "plan", prompts.planning_system(),
            prompts.planning_user(job_description, company, role_title,
                                  [item.model_dump(mode="json") for item in analysis.requirements],
                                  fact_payload, supplemental_payload),
            PlanOutput, config_value("CLAUDE_PLAN_MAX_TOKENS"),
        )
        analysis = merge_llm_job_analysis(analysis, response["analysis"], job_description)
        if not role_title.strip() and response["analysis"]["role_title"].strip():
            analysis.role_title = response["analysis"]["role_title"].strip()[:120]
        mappings = response["mappings"]
    result.job_analysis = analysis
    matrix = build_evidence_matrix(analysis, facts, supplemental, mappings)
    # Original resume assessment excludes supplemental facts that are absent
    # from the original document.
    original_matrix = build_evidence_matrix(analysis, facts, llm_mappings=mappings)
    original_text = facts_to_text(facts)
    result.original_assessment = build_assessment(
        original_matrix, original_text,
        tools_and_tech=analysis.tools_and_tech,
        domain_vocabulary=analysis.domain_vocabulary,
        structure_score=estimate_structure_score(original_text),
        quality_score=estimate_accomplishment_quality(original_text),
    )

    positioning = choose_positioning(analysis, matrix)
    result.positioning = positioning
    optimized, _, gaps, terminology, decisions = deterministic_optimize(
        facts, analysis, matrix, positioning, supplemental,
    )
    questions = [gap.clarification_question for gap in gaps if gap.clarification_question]
    if not result.demo_mode:
        response = model_session.request(
            "rewrite",
            prompts.rewrite_system(),
            prompts.rewrite_user(
                fact_payload,
                {"role": analysis.role_title, "priorities": analysis.hiring_priorities,
                 "tools": analysis.tools_and_tech, "target_internal_score": config_value("TARGET_SCORE"),
                 "outcomes": analysis.deliverables_and_outcomes,
                 "domain_vocabulary": analysis.domain_vocabulary},
                positioning.framing, supplemental_payload,
                [{"req_id": item.req_id, "requirement": item.requirement,
                  "importance": item.importance, "evidence_fact_ids": item.evidence_fact_ids,
                  "evidence_strength": item.evidence_strength.value,
                  "treatment": item.recommended_treatment} for item in matrix],
            ),
            RewriteOutput, config_value("CLAUDE_REWRITE_MAX_TOKENS"),
        )
        if not response["experience"] and facts.experience:
            raise LLMError("Claude returned no experience entries. Please try again.")
        optimized = merge_llm_optimized(optimized, response, facts, supplemental)
        decisions.extend(response.get("content_decisions", []))
        questions.extend(response.get("clarification_questions", []))
        decisions.append("Claude rewrote the resume using source facts with explicit evidence references.")

    optimized, audit = audit_optimized_resume(optimized, facts, supplemental)
    questions.extend(audit.clarification_questions)
    result.warnings.extend(audit.removed_claims[:12])
    if audit.removed_claims:
        decisions.append("Removed claims that failed the factual audit.")

    # Keep the audited Claude rewrite. Layout fitting changes spacing and omits
    # lower-priority content; it never replaces the rewrite with demo wording.
    optimized, latex, compilation, layout_decisions = fit_resume_to_one_page(
        optimized, page_size=page_size,
    )
    decisions.extend(layout_decisions)
    result.optimized_resume = optimized
    result.latex_source = latex
    result.compilation = compilation
    result.passes_used = 1
    if not compilation.success:
        result.errors.append(compilation.error or "A verified one-page PDF could not be produced.")
    result.warnings.extend(compilation.warnings)

    final_text = optimized_to_text(optimized)
    final_matrix = rematch_matrix_for_text(
        matrix, final_text, facts=facts, optimized=optimized, supplemental=supplemental,
    )
    result.optimized_assessment = build_assessment(
        final_matrix, final_text,
        tools_and_tech=analysis.tools_and_tech,
        domain_vocabulary=analysis.domain_vocabulary,
        structure_score=estimate_structure_score(final_text),
        quality_score=estimate_accomplishment_quality(final_text),
    )
    target = config_value("TARGET_SCORE")
    # One bounded refinement closes presentation gaps; source gaps cannot be repaired
    # by another generation. Score the audited, fitted candidate before accepting it.
    supported_gaps = any(before.current_coverage > after.current_coverage
                         for before, after in zip(matrix, final_matrix))
    can_refine = any(row.current_coverage > 0 for row in matrix) and (
        supported_gaps or estimate_accomplishment_quality(final_text) < 0.85)
    if (not result.demo_mode and result.optimized_assessment.score.total < target
            and config_value("MAX_OPTIMIZATION_PASSES") > 1 and can_refine):
        try:
            response = model_session.request(
                "refine", prompts.rewrite_system(),
                prompts.repair_user(
                    fact_payload, supplemental_payload,
                    optimized.model_dump(mode="json", exclude_defaults=True),
                    {"score": result.optimized_assessment.score.total,
                     "requirements": [{"req_id": row.req_id, "requirement": row.requirement,
                                       "coverage": row.current_coverage,
                                       "supported_coverage": matrix[index].current_coverage,
                                       "evidence_fact_ids": matrix[index].evidence_fact_ids}
                                      for index, row in enumerate(final_matrix)]},
                    analysis.hiring_priorities, audit.removed_claims, target,
                ), RewriteOutput, config_value("CLAUDE_REPAIR_MAX_TOKENS"),
            )
            result.passes_used = 2
            candidate = merge_llm_optimized(optimized.model_copy(deep=True), response, facts, supplemental)
            candidate, candidate_audit = audit_optimized_resume(candidate, facts, supplemental)
            candidate, candidate_tex, candidate_pdf, candidate_decisions = fit_resume_to_one_page(
                candidate, page_size=page_size,
            )
            candidate_text = optimized_to_text(candidate)
            candidate_matrix = rematch_matrix_for_text(
                matrix, candidate_text, facts=facts, optimized=candidate, supplemental=supplemental,
            )
            assessment = build_assessment(candidate_matrix, candidate_text,
                                          tools_and_tech=analysis.tools_and_tech,
                                          domain_vocabulary=analysis.domain_vocabulary)
            if (assessment.score.total > result.optimized_assessment.score.total
                    and not (set(assessment.score.mandatory_gaps)
                             - set(result.optimized_assessment.score.mandatory_gaps))
                    and (candidate_pdf.success or not compilation.success)):
                optimized, latex, compilation = candidate, candidate_tex, candidate_pdf
                final_text, final_matrix = candidate_text, candidate_matrix
                result.optimized_resume, result.latex_source = optimized, latex
                result.compilation, result.optimized_assessment = compilation, assessment
                result.errors = [] if compilation.success else [compilation.error]
                result.warnings.extend(candidate_audit.removed_claims[:12] + compilation.warnings)
                decisions.extend(candidate_decisions + response.get("content_decisions", []))
                questions.extend(candidate_audit.clarification_questions + response.get("clarification_questions", []))
                decisions.append("Accepted the refinement after its audited final content improved the match estimate.")
            else:
                decisions.append("Kept the first resume because refinement did not improve verified coverage and score.")
        except LLMError:
            result.warnings.append("The optional refinement was unavailable. The first audited resume is retained.")

    result.target_reached = (result.optimized_assessment.score.total >= target
                             and not result.optimized_assessment.score.mandatory_gaps)
    if result.target_reached:
        result.score_ceiling_reason = (
            f"Internal match estimate reached {target}+ using evidence-backed coverage. "
            "This is not an employer ATS score."
        )
    else:
        result.score_ceiling_reason = (
            f"Internal match estimate is {result.optimized_assessment.score.total}/100. "
            "Stronger evidence is needed to increase coverage without inventing qualifications."
        )
        if result.optimized_assessment.score.mandatory_gaps:
            result.score_ceiling_reason += " Mandatory qualification gaps remain."

    # Changes and recommendations describe the final PDF content after fitting.
    originals = {
        bullet.fact_id: bullet.text
        for entry in [*facts.experience, *facts.projects] for bullet in entry.bullets
    }
    changes = []
    for entry in [*optimized.experience, *optimized.projects]:
        for bullet in entry.bullets:
            before = " ".join(originals[fid] for fid in bullet.fact_ids if fid in originals)
            if before and before != bullet.text:
                changes.append(BulletChange(
                    before=before, after=bullet.text,
                    reason="Tailored wording to the role using cited source evidence.",
                    fact_ids=bullet.fact_ids,
                ))
    result.change_log = changes
    result.gaps = _final_gaps(final_matrix)
    result.terminology_added = [
        item for item in terminology
        if str(item.get("term", "")).lower() in final_text.lower()
    ]
    questions.extend(gap.clarification_question for gap in result.gaps if gap.clarification_question)
    result.clarification_questions = _unique(questions)
    result.content_decisions = _unique(decisions)
    result.recommendations, result.next_actions = build_recommendations(
        analysis, result.gaps, changes, result.score_ceiling_reason, result.clarification_questions,
    )
    result.token_usage = model_session.usage()
    # Persist every completed run, including demo runs and compiler diagnostics.
    if knowledge_store is not None:
        try:
            result.run_id = knowledge_store.save_run(
                result.to_template_dict(), job_description=job_description,
                company=company, role_title=analysis.role_title or role_title,
                profile_id=profile_id, resume_hash=hashlib.sha256(resume_latex.encode()).hexdigest(),
            )
        except (sqlite3.Error, OSError, ValueError):
            result.warnings.append("This result could not be saved to local history. You can still download the resume.")
    return result


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item and item.strip()))


def _final_gaps(matrix) -> list[GapItem]:
    """Report gaps from the actual retained content after one-page fitting."""
    gaps = []
    for row in matrix:
        if row.evidence_strength == EvidenceStrength.NONE or row.current_coverage == 0:
            gaps.append(GapItem(
                requirement=row.requirement, classification=row.classification,
                gap_type="missing",
                recommendation="This qualification is not evidenced in the final resume. Add confirmed supporting facts before claiming it.",
            ))
        elif row.current_coverage < 0.5:
            gaps.append(GapItem(
                requirement=row.requirement, classification=row.classification,
                gap_type="needs_clarification",
                recommendation="The final resume offers partial evidence. Confirm scope and your contribution before strengthening the claim.",
                clarification_question=f"What verified experience supports: {row.requirement[:120]}?",
            ))
    return gaps

