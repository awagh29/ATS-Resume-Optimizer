"""Parse LaTeX resume into structured ResumeFacts (deterministic + optional LLM enrich)."""

from __future__ import annotations

import re
from typing import Any

from app.models.schemas import (
    ContactInfo,
    EducationItem,
    ExperienceBullet,
    ExperienceRole,
    ProjectItem,
    ResumeFacts,
    SkillGroup,
)


def _expand_data_macro(text: str, command: str, argument_count: int, render) -> str:
    """Expand a known macro with balanced arguments, including nested formatting."""
    pattern = re.compile(r"\\" + re.escape(command) + r"\b")
    output: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() < cursor:
            continue
        pos = match.end()
        arguments = []
        for _ in range(argument_count):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text) or text[pos] != "{":
                break
            start = pos + 1
            pos += 1
            depth = 1
            while pos < len(text) and depth:
                if text[pos] == "\\" and pos + 1 < len(text) and text[pos + 1] in "{}":
                    pos += 2
                    continue
                if text[pos] == "{":
                    depth += 1
                elif text[pos] == "}":
                    depth -= 1
                pos += 1
            if depth:
                break
            arguments.append(text[start:pos - 1])
        if len(arguments) == argument_count:
            output.extend([text[cursor:match.start()], render(arguments)])
            cursor = pos
    output.append(text[cursor:])
    return "".join(output)


def _strip_latex_commands(text: str) -> str:
    """Best-effort plain text from LaTeX for extraction."""
    t = text
    # The preamble contains package names, dimensions and macro definitions, not facts.
    if r"\begin{document}" in t:
        t = t.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    # Remove comments
    t = re.sub(r"(?<!\\)%.*?$", "", t, flags=re.M)
    # Common Jake's-resume macros. Expand the data arguments before dropping commands.
    t = _expand_data_macro(t, "resumeSubheading", 4, lambda args: f"\n\n{args[2]} at {args[0]} | {args[3]} {args[1]}\n")
    t = _expand_data_macro(t, "resumeProjectHeading", 2, lambda args: f"\n\n{args[0]} {args[1]}\n")
    t = _expand_data_macro(t, "resumeItem", 1, lambda args: "\\item " + args[0])
    t = _expand_data_macro(t, "cventry", 6, lambda args: f"\n\n{args[1]} at {args[2]} | {args[3]} {args[0]}\n" + (args[5] if r"\item" in args[5] else "\\item " + args[5]) + "\n")
    t = _expand_data_macro(t, "cvitem", 2, lambda args: f"\n{args[0]}: {args[1]}\n")
    # Common resume macros → keep args
    replacements = [
        (r"\\textbf\{([^{}]*)\}", r"\1"),
        (r"\\textit\{([^{}]*)\}", r"\1"),
        (r"\\emph\{([^{}]*)\}", r"\1"),
        (r"\\underline\{([^{}]*)\}", r"\1"),
        (r"\\href\{[^{}]*\}\{([^{}]*)\}", r"\1"),
        (r"\\url\{([^{}]*)\}", r"\1"),
        (r"\\section\*?\{([^{}]*)\}", r"\nSECTION:\1\n"),
        (r"\\subsection\*?\{([^{}]*)\}", r"\nSUBSECTION:\1\n"),
        (r"(?m)^[ \t]*\\item\b[ \t]*", "- "),
        (r"\\item\b", r"\n- "),
        (r"\\\\(?:\[[^\]]*\])?[ \t]*(?:\r?\n)?", r"\n"),
        (r"\\&", "&"),
        (r"\\%", "%"),
        (r"\\_", "_"),
        (r"\\\$", "$"),
        (r"~", " "),
    ]
    for pat, repl in replacements:
        t = re.sub(pat, repl, t)
    t = re.sub(r"(?m)^[ \t]*\\(?:begin|end)\{[^{}]*\}(?:\[[^\]]*\])?[ \t]*\r?\n", "", t)
    t = re.sub(r"\\(?:begin|end)\{[^{}]*\}(?:\[[^\]]*\])?", "", t)
    # Font switches and spacing commands have no text arguments to preserve.
    t = re.sub(r"\\(?:hfill|centering|small|large|Large|LARGE|Huge|huge|bfseries|itshape|normalfont)\b", " ", t)
    t = re.sub(r"\\(?:vspace|hspace)\*?\{[^{}]*\}", " ", t)
    # Drop remaining simple commands
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", t)
    t = re.sub(r"[{}]", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _next_id(prefix: str, counter: list[int]) -> str:
    counter[0] += 1
    return f"{prefix}{counter[0]}"


def parse_resume_latex(latex: str) -> ResumeFacts:
    """
    Deterministic LaTeX → ResumeFacts extraction.

    Supports common patterns (moderncv-ish, article+itemize, simple section resumes).
    Reports limitations in extraction_notes.
    """
    facts = ResumeFacts()
    notes: list[str] = []
    counter = [0]

    if not latex or not latex.strip():
        facts.extraction_notes.append("Empty resume input.")
        return facts

    plain = _strip_latex_commands(latex)
    facts.contact = _extract_contact(latex, plain)

    # Section split
    sections = _split_sections(plain)
    if not sections:
        notes.append("Could not detect \\section headings; used whole-document heuristics.")
        sections = {"body": plain}

    # Map section names
    for name, body in sections.items():
        key = name.lower().strip()
        if any(k in key for k in ("experience", "employment", "work history", "professional")):
            facts.experience.extend(_parse_experience(body, counter))
        elif any(k in key for k in ("education", "academic")):
            facts.education.extend(_parse_education(body, counter))
        elif any(k in key for k in ("project",)):
            facts.projects.extend(_parse_projects(body, counter))
        elif any(k in key for k in ("skill", "technolog", "tools")):
            facts.skills.extend(_parse_skills(body, counter))
        elif any(k in key for k in ("summary", "profile", "objective", "about")):
            facts.summary = re.sub(r"\s+", " ", body).strip()[:800]
        elif any(k in key for k in ("certif", "license")):
            facts.certifications = [
                re.sub(r"^[-•*\s]+", "", ln).strip()
                for ln in body.splitlines()
                if ln.strip()
            ]
        elif key not in ("body",):
            lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
            if lines:
                facts.other_sections[name] = lines[:30]

    if not facts.experience:
        # Heuristic: look for "Title at Company" or "Company — Title" patterns
        guessed = _heuristic_experience(plain, counter)
        if guessed:
            facts.experience = guessed
            notes.append("Experience inferred from date/title patterns (no Experience section found).")
        else:
            notes.append("No experience section extracted; optimization may be limited.")

    if not facts.skills:
        # Pull comma-separated tech-looking tokens from a Skills-like block in raw latex
        skill_match = re.search(
            r"(?:skills|technologies)\s*[:\n](.+?)(?:\n\s*\n|SECTION:|$)",
            plain,
            flags=re.I | re.S,
        )
        if skill_match:
            facts.skills = _parse_skills(skill_match.group(1), counter)

    if not facts.contact.name:
        # First non-empty line often is the name in resumes
        for ln in plain.split("SECTION:", 1)[0].splitlines():
            ln = ln.strip()
            if ln and "@" not in ln and len(ln.split()) <= 5 and len(ln) < 60 and re.search(r"[A-Za-z]", ln):
                if not ln.lower().startswith("section:"):
                    facts.contact.name = ln
                    notes.append("Name inferred from first heading-like line.")
                    break

    facts.extraction_notes = notes
    facts.collect_fact_ids()
    return facts


def _extract_contact(latex: str, plain: str) -> ContactInfo:
    c = ContactInfo()
    email = re.search(r"[\w.+-]+@[\w.-]+\.\w+", plain)
    if email:
        c.email = email.group(0)
    phone = re.search(
        r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}",
        plain,
    )
    if phone:
        c.phone = phone.group(0).strip()
    li = re.search(r"(?:linkedin\.com/in/[\w\-/%]+)", plain, re.I)
    if li:
        c.linkedin = "https://" + li.group(0)
    gh = re.search(r"(?:github\.com/[\w\-]+)", plain, re.I)
    if gh:
        c.github = "https://" + gh.group(0)

    # moderncv \name{First}{Last}
    name_mc = re.search(r"\\name\{([^{}]*)\}\{([^{}]*)\}", latex)
    if name_mc:
        c.name = f"{name_mc.group(1).strip()} {name_mc.group(2).strip()}".strip()
    name_cmd = re.search(r"\\namelarge\{([^{}]*)\}", latex) or re.search(
        r"\\newcommand\{\\name\}\{([^{}]*)\}", latex
    )
    if not c.name and name_cmd:
        c.name = name_cmd.group(1).strip()

    loc = re.search(
        r"\b([A-Z][a-zA-Z .]+,\s*[A-Z]{2}(?:\s+\d{5})?)\b",
        plain,
    )
    if loc:
        c.location = loc.group(1).strip()
    return c


def _split_sections(plain: str) -> dict[str, str]:
    parts = re.split(r"(?:^|\n)\s*SECTION:\s*", plain)
    if len(parts) <= 1:
        return {}
    sections: dict[str, str] = {}
    # first chunk before any section may be header
    if parts[0].strip():
        sections["Header"] = parts[0].strip()
    for chunk in parts[1:]:
        lines = chunk.splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        sections[title] = body
    return sections


def _parse_bullets(body: str, counter: list[int], prefix: str) -> list[ExperienceBullet]:
    bullets: list[ExperienceBullet] = []
    for ln in body.splitlines():
        ln = ln.strip()
        if ln.startswith(("-", "•", "*")):
            text = re.sub(r"^[-•*\s]+", "", ln).strip()
            if text:
                bullets.append(
                    ExperienceBullet(fact_id=_next_id(prefix, counter), text=text, source="resume")
                )
        elif bullets and ln and not ln.startswith(("SECTION:", "SUBSECTION:")):
            # A LaTeX item can wrap across source lines; retain the complete claim.
            bullets[-1].text += " " + ln
    return bullets


def _parse_experience(body: str, counter: list[int]) -> list[ExperienceRole]:
    roles: list[ExperienceRole] = []
    date_re = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})"
        r"\s*(?:[–—-]+|\bto\b)\s*"
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4}|Present|Current)",
        re.I,
    )
    # A new title/date header after bullets also starts a role, even when the
    # author did not add a blank source line between itemize environments.
    separated = []
    seen_bullet = False
    for line in body.splitlines():
        if seen_bullet and not line.lstrip().startswith(("-", "•", "*")) and date_re.search(line):
            separated.append("")
            seen_bullet = False
        separated.append(line)
        seen_bullet = seen_bullet or line.lstrip().startswith(("-", "•", "*"))
    blocks = re.split(r"\n\s*\n", "\n".join(separated))
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if lines[0].startswith(("-", "•", "*")) and roles:
            roles[-1].bullets.extend(_parse_bullets(block, counter, "b"))
            continue
        header = lines[0]
        dates = date_re.search(block)
        start = dates.group(1) if dates else ""
        end = dates.group(2) if dates else ""
        # Dates often sit on the same line as the title after LaTeX stripping
        header_clean = date_re.sub("", header).strip(" |·•-\t")
        header_clean = re.sub(r"\s{2,}", " ", header_clean).strip()
        employer, title, location = _split_role_header(header_clean or header)
        header_lines = 1
        if not employer and len(lines) > 1 and not lines[1].startswith(("-", "•", "*")):
            # The common article layout puts the official title/date on line one
            # and employer/location on line two.
            employer_line = date_re.sub("", lines[1]).strip(" |·•-\t")
            company_parts = employer_line.split(",", 1)
            employer = company_parts[0].strip()
            location = company_parts[1].strip() if len(company_parts) > 1 else ""
            header_lines = 2
        bullet_src = "\n".join(lines[header_lines:]) if len(lines) > header_lines else ""
        bullets = _parse_bullets(bullet_src, counter, "b")
        if not employer and not title and not bullets:
            continue
        roles.append(
            ExperienceRole(
                fact_id=_next_id("exp", counter),
                employer=employer,
                title=title or header[:80],
                location=location,
                start_date=start,
                end_date=end,
                bullets=bullets,
            )
        )
    return roles


def _split_role_header(header: str) -> tuple[str, str, str]:
    header = re.sub(r"\s+", " ", header).strip()
    # Title at Company | Location
    m = re.match(
        r"(.+?)\s+at\s+(.+?)(?:\s*[|•]\s*(.+))?$",
        header,
        re.I,
    )
    if m:
        return m.group(2).strip(), m.group(1).strip(), (m.group(3) or "").strip()
    # Company — Title
    m = re.match(r"(.+?)\s+(?:[—|]|-+)\s+(.+?)(?:\s*[|•]\s*(.+))?$", header)
    if m:
        left, right = m.group(1).strip(), m.group(2).strip()
        # Heuristic: if left looks like title (has Engineer/Manager), swap
        title_words = ("engineer", "developer", "manager", "analyst", "intern", "scientist", "lead")
        if any(w in left.lower() for w in title_words):
            return right, left, (m.group(3) or "").strip()
        return left, right, (m.group(3) or "").strip()
    return "", header, ""


def _parse_education(body: str, counter: list[int]) -> list[EducationItem]:
    items: list[EducationItem] = []
    blocks = re.split(r"\n\s*\n", body)
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        degree = ""
        field = ""
        institution = re.sub(r"\s+(?:19|20)\d{2}\s*$", "", lines[0]).strip()
        grad = ""
        year = re.search(r"(20\d{2}|19\d{2})", block)
        if year:
            grad = year.group(1)
        deg = re.search(
            r"(Bachelor|Master|B\.S\.|M\.S\.|B\.A\.|M\.A\.|Ph\.?D\.?|Associate)[^,\n]*",
            block,
            re.I,
        )
        if deg:
            degree = deg.group(0).strip()
        items.append(
            EducationItem(
                fact_id=_next_id("edu", counter),
                institution=institution[:120],
                degree=degree,
                field=field,
                graduation_date=grad,
                details=[ln for ln in lines[1:] if not ln.startswith("-") and ln.strip() not in {degree, field, grad}][:5],
            )
        )
    return items


def _parse_projects(body: str, counter: list[int]) -> list[ProjectItem]:
    projects: list[ProjectItem] = []
    blocks = re.split(r"\n\s*\n", body)
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        name = re.sub(r"^[-•*\s]+", "", lines[0])[:100]
        bullets = _parse_bullets(block, counter, "pb")
        techs: list[str] = []
        tech_line = re.search(r"(?:Technologies|Stack|Tools)\s*:\s*(.+)", block, re.I)
        if tech_line:
            techs = [t.strip() for t in re.split(r"[,;/]", tech_line.group(1)) if t.strip()]
        projects.append(
            ProjectItem(
                fact_id=_next_id("proj", counter),
                name=name,
                bullets=bullets,
                technologies=techs,
            )
        )
    return projects


def _parse_skills(body: str, counter: list[int]) -> list[SkillGroup]:
    groups: list[SkillGroup] = []
    for ln in body.splitlines():
        ln = ln.strip().lstrip("-•* ").strip()
        if not ln:
            continue
        if ":" in ln:
            cat, rest = ln.split(":", 1)
            skills = _split_skill_values(rest)
            groups.append(
                SkillGroup(fact_id=_next_id("sk", counter), category=cat.strip(), skills=skills)
            )
        else:
            skills = _split_skill_values(ln)
            if skills:
                groups.append(
                    SkillGroup(
                        fact_id=_next_id("sk", counter),
                        category="Skills",
                        skills=skills,
                    )
                )
    return groups


def _split_skill_values(text: str) -> list[str]:
    """Keep details such as AWS (ECS, S3) together as one supported skill."""
    result = []
    start = depth = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char in ",;|" and depth == 0:
            if text[start:i].strip():
                result.append(text[start:i].strip())
            start = i + 1
    if text[start:].strip():
        result.append(text[start:].strip())
    return result


def _heuristic_experience(plain: str, counter: list[int]) -> list[ExperienceRole]:
    roles: list[ExperienceRole] = []
    date_re = re.compile(
        r"(.{10,80}?)\n.*?((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})"
        r"\s*[–\-—to]+\s*"
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4}|Present|Current)",
        re.I | re.S,
    )
    for m in date_re.finditer(plain):
        header = m.group(1).strip().splitlines()[-1]
        employer, title, location = _split_role_header(header)
        roles.append(
            ExperienceRole(
                fact_id=_next_id("exp", counter),
                employer=employer or "Unknown Employer",
                title=title or header[:80],
                location=location,
                start_date=m.group(2),
                end_date=m.group(3),
                bullets=[],
            )
        )
        if len(roles) >= 6:
            break
    return roles


def merge_llm_facts(base: ResumeFacts, llm_data: dict[str, Any]) -> ResumeFacts:
    """Optionally enrich deterministic parse with validated LLM fields (non-destructive)."""
    if not llm_data:
        return base
    # Only fill empty fields; never invent over existing
    contact = llm_data.get("contact") or {}
    if isinstance(contact, dict):
        for field in ("name", "email", "phone", "location", "linkedin", "github", "website"):
            if not getattr(base.contact, field) and contact.get(field):
                setattr(base.contact, field, str(contact[field])[:200])
    if not base.summary and llm_data.get("summary"):
        base.summary = str(llm_data["summary"])[:800]
    base.collect_fact_ids()
    return base
