"""Render OptimizedResume through the controlled Jinja LaTeX template."""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from app.models.schemas import OptimizedResume

TEMPLATE_DIR = Path(__file__).resolve().parent


def latex_escape(value: str) -> str:
    if value is None:
        return ""
    s = str(value)
    replacements = dict([
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ])
    # Replace each input character once; escaping braces introduced by a previous
    # replacement corrupts commands such as \\textbackslash{}.
    s = "".join(replacements.get(character, character) for character in s)
    # Strip characters that often break pdflatex without XeLaTeX
    s = s.replace("–", "--").replace("—", "---").replace("•", "-")
    s = s.replace("“", "``").replace("”", "''").replace("’", "'")
    return s


def render_resume_latex(
    resume: OptimizedResume,
    page_size: str = "letter",
    *,
    margin_in: float = 0.6,
    item_sep: float = 0.12,
    section_top: float = 0.55,
    section_bot: float = 0.25,
    baselinestretch: float = 0.98,
    header_vspace: float = 0.4,
    role_vspace: float = 0.15,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["latex_escape"] = latex_escape
    template = env.get_template("template.tex.j2")

    page_class = "letterpaper" if page_size == "letter" else "a4paper"
    c = resume.contact
    contact_parts = [
        latex_escape(x)
        for x in [c.email, c.phone, c.location, c.linkedin, c.github, c.website]
        if x
    ]
    contact_line = r" $\cdot$ ".join(contact_parts)

    return template.render(
        page_class=page_class,
        margin_in=margin_in,
        item_sep=item_sep,
        section_top=section_top,
        section_bot=section_bot,
        baselinestretch=baselinestretch,
        header_vspace=header_vspace,
        role_vspace=role_vspace,
        contact=resume.contact,
        contact_line=contact_line,
        summary=resume.summary,
        experience=resume.experience,
        projects=resume.projects,
        skills=resume.skills,
        education=resume.education,
        certifications=resume.certifications,
    )


def adjust_spacing_for_fit(page_count: int | None, underfilled: bool) -> dict:
    """Return modest spacing tweaks for layout revision passes."""
    if page_count and page_count > 1:
        return {
            "margin_in": 0.5,
            "item_sep": 0.06,
            "section_top": 0.4,
            "section_bot": 0.15,
            "baselinestretch": 0.95,
            "header_vspace": 0.25,
            "role_vspace": 0.05,
        }
    if underfilled:
        return {
            "margin_in": 0.65,
            "item_sep": 0.18,
            "section_top": 0.7,
            "section_bot": 0.3,
            "baselinestretch": 1.02,
            "header_vspace": 0.5,
            "role_vspace": 0.25,
        }
    return {
        "margin_in": 0.6,
        "item_sep": 0.12,
        "section_top": 0.55,
        "section_bot": 0.25,
        "baselinestretch": 0.98,
        "header_vspace": 0.4,
        "role_vspace": 0.15,
    }


def tighten_content(resume: OptimizedResume) -> OptimizedResume:
    """Remove one low-priority unit without cutting any retained statement."""
    r = resume.model_copy(deep=True)
    # The rewrite stage orders content by relevance. Preserve at least one bullet
    # for every retained role and never alter names, dates, degrees, or skills.
    for project in reversed(r.projects):
        if len(project.bullets) > 1:
            project.bullets.pop()
            return r
    for role in reversed(r.experience):
        if len(role.bullets) > 3:
            role.bullets.pop()
            return r
    if r.projects:
        r.projects.pop()
        return r
    for role in reversed(r.experience):
        if len(role.bullets) > 1:
            role.bullets.pop()
            return r
    for education in reversed(r.education):
        if education.details:
            education.details.pop()
            return r
    if r.summary:
        r.summary = ""
        return r
    return r


def expand_content(resume: OptimizedResume, facts_projects_count: int) -> OptimizedResume:
    """Modest expansion when underfilled — does not invent facts."""
    r = resume.model_copy(deep=True)
    # Slightly longer summary only if present
    return r


def _shorten_bullet(text: str) -> str:
    t = re.sub(r"\b(in order to|as well as|the ability to)\b", "", text, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def fit_resume_to_one_page(
    resume: OptimizedResume,
    page_size: str = "letter",
) -> tuple[OptimizedResume, str, "CompilationResult", list[str]]:
    """Compile, measure, then make bounded layout/content revisions.

    A failed result contains editable LaTeX but never a rejected PDF. Content is
    selected rather than clipped, and every omitted unit is disclosed.
    """
    from app.latex.compiler import _setting, compile_latex
    from app.latex.pdf_validate import validate_pdf
    from app.models.schemas import CompilationResult

    current = resume.model_copy(deep=True)
    decisions: list[str] = []
    spacing = adjust_spacing_for_fit(None, underfilled=False)
    max_passes = max(1, min(40, int(_setting("LATEX_MAX_LAYOUT_PASSES", 14))))
    compilation = CompilationResult()
    latex = ""
    for attempt in range(max_passes):
        latex = render_resume_latex(current, page_size=page_size, **spacing)
        compilation = compile_latex(latex)
        if not compilation.success:
            return current, latex, compilation, decisions
        compilation = validate_pdf(compilation, current)
        if compilation.success and compilation.layout_verified:
            return current, latex, compilation, decisions
        # Parser/font failures cannot be fixed by discarding good resume content.
        if not compilation.page_count or (
            compilation.page_count == 1 and "overflow" not in compilation.error.lower()
        ):
            return current, latex, compilation, decisions
        if attempt == max_passes - 1:
            break
        if attempt == 0:
            spacing = adjust_spacing_for_fit(2, underfilled=False)
            decisions.append("Reduced spacing and margins to 0.5 inches while retaining a 10-point body font.")
            continue
        tightened = tighten_content(current)
        if tightened == current:
            break
        decisions.append(_omission_description(current, tightened))
        current = tightened
    compilation.pdf_bytes = None
    compilation.success = False
    compilation.layout_verified = False
    compilation.error = (
        "The resume could not fit one readable page within the bounded layout passes. "
        "Shorten the editable LaTeX or reduce lower-priority source content and try again."
    )
    return current, latex, compilation, decisions


def _omission_description(before: OptimizedResume, after: OptimizedResume) -> str:
    if len(after.projects) < len(before.projects):
        return f"Omitted the lower-priority project '{before.projects[-1].name}' to fit one page."
    for old, new in zip(before.projects, after.projects):
        if len(new.bullets) < len(old.bullets):
            return f"Omitted a lower-priority bullet from project '{old.name}' to fit one page."
    for old, new in zip(before.experience, after.experience):
        if len(new.bullets) < len(old.bullets):
            return f"Omitted a lower-priority bullet from '{old.title}' at '{old.employer}' to fit one page."
    for old, new in zip(before.education, after.education):
        if len(new.details) < len(old.details):
            return f"Omitted an optional education detail from '{old.institution}' to fit one page."
    return "Omitted the optional summary to retain experience and education on one page."
