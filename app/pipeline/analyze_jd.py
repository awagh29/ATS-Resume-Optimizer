"""Job description analysis — deterministic baseline + LLM enrichment."""

from __future__ import annotations

import re
from typing import Any

from app.models.schemas import Classification, JobAnalysis, RequirementItem
from app.pipeline.scoring import term_present


# Headings must match as headings. A bullet containing "required" is content,
# and must never disappear merely because it includes a classification word.
SECTION_PATTERNS = [
    ("required", re.compile(r"(?:(?:basic|minimum|required|essential|mandatory|logistical|additional)\s+(?:qualifications?|requirements?|skills?)|qualifications?\s+required|must[- ]haves?|required)", re.I)),
    ("preferred", re.compile(r"(?:(?:preferred|desired)\s+(?:qualifications?|requirements?|skills?)|preferred|nice[- ]to[- ]haves?|bonus(?:\s+points)?|a\s+plus)", re.I)),
    ("responsibilities", re.compile(r"(?:responsibilities|key\s+responsibilities|what\s+you[’']?ll\s+do|what\s+you\s+will\s+do|the\s+role|about\s+the\s+(?:role|job)|your\s+(?:role|impact)|duties|you\s+will)", re.I)),
    ("requirements", re.compile(r"(?:requirements?|qualifications?|what\s+you\s+(?:bring|have)|you\s+(?:bring|have)|who\s+you\s+are|about\s+you)", re.I)),
    ("general", re.compile(r"(?:benefits|what\s+we\s+offer|compensation|salary|equal\s+opportunity|about\s+(?:us|the\s+company)|our\s+company|how\s+to\s+apply)", re.I)),
]
_BULLET = re.compile(r"^(?:[-•*▪◦]\s*|\d+[.)]\s+)")
_ELIGIBILITY = re.compile(r"\b(?:clearance|citizen(?:ship)?|authori[sz](?:ed|ation)|visa|sponsorship|remote|hybrid|on[- ]?site|relocat\w*|travel|work\s+permit)\b", re.I)
_ACTION = re.compile(r"\b(?:design|build|develop|implement|operate|maintain|deliver|own|lead|mentor|collaborat\w*|partner|improve|manage|analy[sz]e|support|create|test|monitor|drive|launch|scale)\b", re.I)


def analyze_job_description(
    jd_text: str,
    company: str = "",
    role_title: str = "",
) -> JobAnalysis:
    """Deterministic JD analysis used as baseline and demo-mode core."""
    analysis = JobAnalysis(
        role_title=role_title.strip(),
        company_hint=company.strip(),
    )
    text = jd_text or ""
    if not text.strip():
        analysis.inferred_notes.append("Empty job description.")
        return analysis

    if not analysis.role_title:
        analysis.role_title = _guess_title(text)
    analysis.seniority = _guess_seniority(text, analysis.role_title)

    sections = _split_jd_sections(text)
    units = [item for items in sections.values() for item in items]
    req_bullets = sections["required"] + sections["requirements"]
    pref_bullets = sections["preferred"] + [b for b in units if _looks_preferred(b)]
    # Explicit preference within a requirements section remains a preference.
    req_bullets = [b for b in req_bullets if not _looks_preferred(b)]
    req_bullets += [b for b in units if _looks_required(b) and not _looks_preferred(b)]
    resp_bullets = [b for b in sections["responsibilities"] if not _looks_preferred(b)]
    resp_bullets += [b for b in sections["general"] if _ACTION.search(b) and not _looks_required(b) and not _looks_preferred(b)]

    analysis.primary_responsibilities = _unique(resp_bullets)
    analysis.required_qualifications = _unique(req_bullets)
    analysis.preferred_qualifications = _unique(pref_bullets)
    analysis.tools_and_tech = _extract_tech(text)
    analysis.domain_vocabulary = _extract_domain_vocab(text, analysis.tools_and_tech)
    analysis.deliverables_and_outcomes = _extract_outcomes(text)
    analysis.leadership_collaboration = [
        _clean(b)
        for b in units
        if re.search(r"\b(?:lead\w*|mentor\w*|collaborat\w*|cross-functional|stakeholder\w*|communicat\w*)\b", b, re.I)
    ]
    analysis.leadership_collaboration = _unique(analysis.leadership_collaboration)
    analysis.eligibility_constraints = _unique(b for b in units if _ELIGIBILITY.search(b))
    analysis.repeated_themes = _repeated_themes(text)
    analysis.hiring_priorities = _hiring_priorities(analysis)
    analysis.inferred_notes = [
        "Seniority and hiring priorities include reasonable inferences; they are not stated employer requirements unless echoed in excerpts.",
    ]
    analysis.requirements = build_requirement_items(analysis, text)
    return analysis


def build_requirement_items(analysis: JobAnalysis, jd_text: str) -> list[RequirementItem]:
    items: list[RequirementItem] = []
    n = 0
    seen: set[str] = set()

    def add(req: str, classification: Classification, importance: float, rationale: str, core: bool = True, citation: str = ""):
        nonlocal n
        if not req.strip() or _key(req) in seen:
            return
        seen.add(_key(req))
        n += 1
        excerpt = _find_excerpt(jd_text, citation or req)
        if not excerpt:
            classification, core = Classification.INFERRED, False
        items.append(
            RequirementItem(
                req_id=f"req{n}",
                requirement=req,
                excerpt=excerpt,
                classification=classification,
                importance=importance,
                rationale=rationale,
                is_core=core,
            )
        )

    for i, r in enumerate(analysis.primary_responsibilities):
        add(
            r,
            Classification.REQUIRED,
            importance=0.9,
            rationale="primary responsibility",
            core=True,
        )
        # tag resp ids separately for scoring convenience
        if items and _key(items[-1].requirement) == _key(r):
            items[-1].req_id = f"resp{i+1}"

    for i, r in enumerate(analysis.required_qualifications):
        add(
            r,
            Classification.REQUIRED,
            importance=0.95,
            rationale="required qualification",
            core=True,
        )

    for i, r in enumerate(analysis.preferred_qualifications):
        add(
            r,
            Classification.PREFERRED,
            importance=0.55,
            rationale="preferred qualification",
            core=False,
        )

    for r in analysis.eligibility_constraints:
        cls = Classification.PREFERRED if _looks_preferred(r) else (
            Classification.REQUIRED if _looks_required(r) or r in analysis.required_qualifications
            else Classification.INFERRED
        )
        add(r, cls, 0.95 if cls == Classification.REQUIRED else 0.5,
            "eligibility or logistical constraint", core=cls == Classification.REQUIRED)

    for t in analysis.tools_and_tech:
        # Avoid duplicating if already in required text
        if any(term_present(t, row.requirement) for row in items):
            continue
        add(
            f"Experience with {t}",
            Classification.PREFERRED if any(term_present(t, item) for item in analysis.preferred_qualifications) else Classification.INFERRED,
            importance=0.45,
            rationale="tool/technology mentioned in JD; required status is not established",
            core=False,
            citation=t,
        )

    for theme in analysis.repeated_themes:
        add(
            f"Demonstrated strength in: {theme}",
            Classification.INFERRED,
            importance=0.55,
            rationale="inferred from repeated JD themes (not an explicit requirement)",
            core=False,
            citation=theme,
        )

    return items


def merge_llm_job_analysis(base: JobAnalysis, llm_data: dict[str, Any], jd_text: str) -> JobAnalysis:
    """Enrich the entire local inventory without dropping explicit constraints.

    Baseline IDs survive enrichment so a combined planning response can map facts
    to the supplied inventory. New IDs are accepted only when unique; citations
    are checked against the original JD before establishing mandatory status.
    """
    if not llm_data:
        return base

    baseline_items = list(base.requirements) or build_requirement_items(base, jd_text)

    def merge_list(attr: str, key: str):
        vals = llm_data.get(key)
        if isinstance(vals, list) and vals:
            cleaned = [str(v).strip() for v in vals if str(v).strip()]
            if cleaned:
                setattr(base, attr, _unique([*getattr(base, attr), *cleaned]))

    for attr, key in [
        ("primary_responsibilities", "primary_responsibilities"),
        ("required_qualifications", "required_qualifications"),
        ("preferred_qualifications", "preferred_qualifications"),
        ("tools_and_tech", "tools_and_tech"),
        ("domain_vocabulary", "domain_vocabulary"),
        ("deliverables_and_outcomes", "deliverables_and_outcomes"),
        ("leadership_collaboration", "leadership_collaboration"),
        ("eligibility_constraints", "eligibility_constraints"),
        ("repeated_themes", "repeated_themes"),
        ("hiring_priorities", "hiring_priorities"),
    ]:
        merge_list(attr, key)

    if llm_data.get("role_title") and not base.role_title:
        base.role_title = str(llm_data["role_title"])[:120]
    if llm_data.get("seniority"):
        base.seniority = str(llm_data["seniority"])[:80]
    if isinstance(llm_data.get("inferred_notes"), list):
        base.inferred_notes = _unique([*base.inferred_notes, *[str(x) for x in llm_data["inferred_notes"]]])

    # Rebuild requirements; if LLM provided structured requirements, validate
    llm_reqs = llm_data.get("requirements")
    if isinstance(llm_reqs, list) and llm_reqs:
        items: list[RequirementItem] = baseline_items
        seen = {_key(item.requirement): item for item in items}
        used_ids = {item.req_id for item in items}
        source_sections = _split_jd_sections(jd_text)
        for raw in llm_reqs:
            if not isinstance(raw, dict):
                continue
            req = str(raw.get("requirement") or "").strip()
            if not req:
                continue
            if _key(req) in seen:
                # The complete deterministic source requirement owns its status
                # and ID; a model cannot downgrade it or replace its citation.
                continue
            cls_raw = str(raw.get("classification") or "required").lower()
            try:
                cls = Classification(cls_raw)
            except ValueError:
                cls = Classification.INFERRED
            try:
                importance = float(raw.get("importance", 0.5))
            except (TypeError, ValueError):
                importance = 0.5
            importance = max(0.0, min(1.0, importance))
            excerpt = str(raw.get("excerpt") or "").strip()
            # A fabricated or unrelated citation must not establish an employer
            # requirement. Require an actual source excerpt with substantive overlap.
            exact_excerpt = _find_excerpt(jd_text, excerpt)
            tokens = _substantive_tokens(req)
            cited_tokens = {token for token in tokens if term_present(token, excerpt)}
            # One generic shared word cannot justify a multi-part qualification.
            quantities = re.findall(r"\d+(?:\.\d+)?\+?", req)
            supported = bool(exact_excerpt and (not tokens or len(cited_tokens) / len(tokens) >= 0.6)
                             and all(quantity in excerpt for quantity in quantities)
                             and (not re.search(r"\byears?\b", req, re.I) or re.search(r"\byears?\b", excerpt, re.I)))
            if not supported:
                excerpt = ""
                cls = Classification.INFERRED
            else:
                excerpt = exact_excerpt
                # A source that expressly says "preferred" cannot become a must.
                if _looks_preferred(excerpt):
                    cls = Classification.PREFERRED
                elif cls == Classification.REQUIRED:
                    source_classes = [
                        section for section, source_items in source_sections.items()
                        for source in source_items if _find_excerpt(source, excerpt)
                    ]
                    if "preferred" in source_classes and not any(section in source_classes for section in ("required", "requirements", "responsibilities")):
                        cls = Classification.PREFERRED
                    elif not (_looks_required(excerpt) or any(section in source_classes for section in ("required", "requirements", "responsibilities"))
                              or any(_looks_required(source) for source in source_sections["general"] if _find_excerpt(source, excerpt))):
                        cls = Classification.INFERRED
            desired_id = str(raw.get("req_id") or "").strip()
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", desired_id) or desired_id in used_ids:
                index = len(items) + 1
                while f"req{index}" in used_ids:
                    index += 1
                desired_id = f"req{index}"
            used_ids.add(desired_id)
            items.append(
                RequirementItem(
                    # IDs are application-owned; duplicate model IDs must never
                    # collapse distinct requirements in subsequent mapping calls.
                    req_id=desired_id,
                    requirement=req,
                    excerpt=excerpt,
                    classification=cls,
                    importance=importance,
                    rationale=str(raw.get("rationale") or "")[:300],
                    is_core=bool(raw.get("is_core", cls == Classification.REQUIRED)) and cls == Classification.REQUIRED,
                )
            )
            seen[_key(req)] = items[-1]
        if items:
            base.requirements = items
            return base

    # Unstructured LLM summaries do not establish new employer requirements.
    # Only supported structured rows above can augment the local inventory.
    base.requirements = baseline_items
    return base


def _guess_title(text: str) -> str:
    first_lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:5]
    for ln in first_lines:
        if 3 < len(ln) < 80 and not ln.lower().startswith("about"):
            if re.search(
                r"(engineer|developer|manager|analyst|designer|scientist|specialist|lead|director)",
                ln,
                re.I,
            ):
                return ln[:100]
    m = re.search(
        r"(Senior|Staff|Principal|Junior|Entry[- ]Level)?\s*"
        r"([A-Z][A-Za-z+/]+(?:\s+[A-Z][A-Za-z+/]+){0,4})\s*"
        r"(Engineer|Developer|Manager|Analyst|Scientist|Designer)",
        text,
    )
    return m.group(0).strip() if m else "Target Role"


def _guess_seniority(text: str, title: str) -> str:
    blob = f"{title}\n{text[:2000]}".lower()
    if re.search(r"\b(principal|staff)\b", blob):
        return "Staff/Principal (inferred from wording)"
    if re.search(r"\b(senior|sr\.?)\b", blob):
        return "Senior (inferred from wording)"
    if re.search(r"\b(junior|jr\.?|entry|new grad|early career)\b", blob):
        return "Junior/Entry (inferred from wording)"
    if re.search(r"\b(lead|manager|director)\b", blob):
        return "Lead/Manager (inferred from wording)"
    return "Mid-level (inferred; not explicitly stated)"


def _extract_bullets(text: str) -> list[str]:
    return [_BULLET.sub("", line.strip()) for line in text.splitlines() if _BULLET.match(line.strip())]


def _sentences(text: str) -> list[str]:
    # Semicolons frequently separate qualifications in pasted one-line JDs.
    # Do not discard short tools or long clauses: both can be decisive.
    parts = re.split(r"\n+|(?<=[.!?;])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _split_jd_sections(text: str) -> dict[str, list[str]]:
    current = "general"
    buckets: dict[str, list[str]] = {k: [] for k, _ in SECTION_PATTERNS}
    combined = "|".join(f"(?:{pattern.pattern})" for _, pattern in SECTION_PATTERNS)
    inline_heading = re.compile(r"(?<!\w)(?:" + combined + r")\s*:", re.I)

    def add(content: str):
        stripped_content = _BULLET.sub("", content.strip()).strip()
        if stripped_content:
            buckets[current].extend(_sentences(stripped_content))

    for line in text.splitlines():
        stripped = line.strip().strip("# ").strip()
        stripped = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
        if not stripped:
            continue
        # Markdown decoration is common in descriptions copied from job boards.
        heading_text = stripped.strip("*_").strip().rstrip(":").strip()
        for key, pattern in SECTION_PATTERNS:
            if not _BULLET.match(stripped) and pattern.fullmatch(heading_text):
                current = key
                break
        else:
            matches = list(inline_heading.finditer(stripped))
            if matches:
                add(stripped[:matches[0].start()])
                for index, match in enumerate(matches):
                    label = match.group(0).rstrip(":").strip()
                    current = next(key for key, pattern in SECTION_PATTERNS if pattern.fullmatch(label))
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
                    add(stripped[match.end():end])
            else:
                add(stripped)
            continue
    return buckets


def _looks_required(b: str) -> bool:
    return bool(re.search(r"\b(?:required|must|minimum|mandatory|need\s+to|years?\s+of|\d+\+?\s+years?)\b", b, re.I))


def _looks_preferred(b: str) -> bool:
    return bool(re.search(r"\b(?:preferred|nice[- ]to[- ]have|bonus|a\s+plus|desirable|desired)\b", b, re.I))


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _key(text: str) -> str:
    return _clean(text).casefold().strip(" .;:•*-")


def _unique(items) -> list[str]:
    seen: set[str] = set()
    output = []
    for item in items:
        clean = _clean(item)
        key = _key(clean)
        if key and key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def _substantive_tokens(text: str) -> set[str]:
    ignored = {"experience", "knowledge", "required", "with", "that", "have", "must", "strong", "ability", "expertise", "proficiency", "working", "familiarity", "years", "professional"}
    return {word for word in re.findall(r"[a-z][a-z0-9+#.-]*", text.lower()) if (len(word) >= 4 or word in {"go", "c#", "c++", "sql", "aws"}) and word not in ignored}


def _extract_tech(text: str) -> list[str]:
    catalog = [
        "Python", "Java", "JavaScript", "TypeScript", "Go", "Golang", "C++", "C#", "Rust",
        "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI", "Spring",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "Spark", "Airflow",
        "SQL", "NoSQL", "GraphQL", "REST", "gRPC", "Linux", "Git",
        "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy",
        "Agile", "Scrum", "Jira", "Figma", "Tableau", "Power BI",
    ]
    found = []
    for term in catalog:
        if term_present(term, text) and (term != "Go" or re.search(r"\bGo\b", text)):
            found.append(term)
    return found


def _extract_domain_vocab(text: str, tech: list[str]) -> list[str]:
    vocab = []
    patterns = [
        r"\b(microservices?|distributed systems?|data pipelines?|ETL|MLOps|DevOps|"
        r"observability|platform engineering|product analytics|A/B testing|"
        r"customer success|go-to-market|compliance|HIPAA|SOC\s?2|security)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            term = m.group(0)
            if term.lower() not in {t.lower() for t in vocab + tech}:
                vocab.append(term)
    return vocab


def _extract_outcomes(text: str) -> list[str]:
    out = []
    for s in _sentences(text):
        if re.search(r"\b(?:deliver|drive|improve|reduce|increase|scale|launch|own)\w*\b", s, re.I):
            out.append(_clean(s))
    return _unique(out)


def _repeated_themes(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
    stop = {
        "with", "that", "this", "from", "your", "have", "will", "work", "team", "role",
        "experience", "years", "ability", "strong", "using", "into", "about", "their",
        "should", "must", "preferred", "requirements", "responsibilities", "including",
        "and", "the", "for", "our", "you", "are", "job", "position",
    }
    counts: dict[str, int] = {}
    for w in words:
        if w in stop:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, c in ranked if c >= 3]


def _hiring_priorities(analysis: JobAnalysis) -> list[str]:
    priorities = []
    if analysis.required_qualifications:
        priorities.append(analysis.required_qualifications[0])
    if analysis.primary_responsibilities:
        priorities.append(analysis.primary_responsibilities[0])
    if analysis.tools_and_tech:
        priorities.append("Hands-on proficiency with: " + ", ".join(analysis.tools_and_tech[:5]))
    if analysis.leadership_collaboration:
        priorities.append(analysis.leadership_collaboration[0])
    if analysis.repeated_themes:
        priorities.append("Repeated emphasis on: " + ", ".join(analysis.repeated_themes[:4]))
    return priorities[:6]


def _find_excerpt(text: str, needle: str) -> str:
    """Return the actual source substring, never an unrelated token match."""
    if not text or not needle:
        return ""
    pattern = r"\s+".join(re.escape(piece) for piece in needle.split())
    match = re.search(pattern, text, re.I)
    return match.group(0) if match else ""
