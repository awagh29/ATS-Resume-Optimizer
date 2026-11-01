"""Accept only a readable, complete, exactly one-page PDF."""

from __future__ import annotations

import unicodedata
from io import BytesIO

from app.models.schemas import CompilationResult, OptimizedResume


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    return "".join(character for character in text if character.isalnum())


def _intended_content(resume: OptimizedResume) -> list[str]:
    content = [resume.contact.name, resume.contact.email, resume.contact.phone,
               resume.contact.location, resume.contact.linkedin, resume.contact.github,
               resume.contact.website, resume.summary]
    for role in resume.experience:
        content.extend([role.title, role.employer, role.start_date, role.end_date])
        content.extend(b.text for b in role.bullets)
    for project in resume.projects:
        content.extend([project.name, *project.technologies])
        content.extend(b.text for b in project.bullets)
    for group in resume.skills:
        content.extend(group.skills)
    for education in resume.education:
        content.extend([education.institution, education.degree, education.field, education.graduation_date])
        content.extend(education.details)
    content.extend(resume.certifications)
    return [x for x in content if x and _normalized(x)]


def _reject(compilation: CompilationResult, reason: str) -> CompilationResult:
    compilation.success = False
    compilation.layout_verified = False
    compilation.pdf_bytes = None
    compilation.error = reason
    if reason not in compilation.warnings:
        compilation.warnings.append(reason)
    return compilation


def validate_pdf(compilation: CompilationResult, resume: OptimizedResume) -> CompilationResult:
    """Fail closed, keeping page_count available for the bounded fitting loop."""
    compilation.layout_verified = False
    if not compilation.success or not compilation.pdf_bytes:
        compilation.pdf_bytes = None
        return compilation
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(compilation.pdf_bytes))
        if reader.is_encrypted:
            return _reject(compilation, "Encrypted PDFs cannot be verified.")
        compilation.page_count = len(reader.pages)
        compilation.extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return _reject(compilation, "PDF could not be parsed; no verified PDF is available.")
    if compilation.page_count != 1:
        return _reject(compilation, f"PDF has {compilation.page_count} pages; exactly one page is required.")
    if len(_normalized(compilation.extracted_text)) < 20:
        return _reject(compilation, "PDF is blank or does not contain readable, extractable resume text.")
    if any("overflows" in warning.lower() for warning in compilation.warnings):
        return _reject(compilation, "PDF contains overflowing text; the layout must be revised.")
    haystack = _normalized(compilation.extracted_text)
    if any(_normalized(item) not in haystack for item in _intended_content(resume)):
        return _reject(compilation, "Some intended resume content is missing from the PDF text; revise the layout.")
    if "\ufffd" in compilation.extracted_text:
        return _reject(compilation, "PDF text contains broken characters; revise the source or font.")
    compilation.error = ""
    compilation.layout_verified = True
    return compilation
