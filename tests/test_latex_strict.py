"""Regression coverage for verified one-page output and restricted compilation."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.latex import compiler
from app.latex.compiler import _restricted_env, compile_latex, validate_latex_source
from app.latex.latex_render import fit_resume_to_one_page, latex_escape, render_resume_latex, tighten_content
from app.latex.pdf_validate import validate_pdf
from app.models.schemas import CompilationResult, ContactInfo, OptimizedBullet, OptimizedResume, OptimizedRole


def make_pdf(text="Alex Rivera Software Engineer Built reliable Python services.", pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def compilation(pdf):
    return CompilationResult(success=True, pdf_bytes=pdf, compiler_available=True)


def test_one_page_complete_resume_is_accepted():
    resume = OptimizedResume(contact=ContactInfo(name="Alex Rivera"))
    result = validate_pdf(compilation(make_pdf()), resume)
    assert result.success and result.layout_verified and result.page_count == 1 and result.pdf_bytes


@pytest.mark.parametrize("pages", [0, 2, 3])
def test_multiple_or_zero_pages_are_never_released(pages):
    result = validate_pdf(compilation(make_pdf(pages=pages)), OptimizedResume())
    assert not result.success and not result.layout_verified and result.pdf_bytes is None
    assert result.page_count == pages


@pytest.mark.parametrize("pdf", [b"not a PDF", make_pdf(text="")])
def test_unreadable_and_blank_pdf_are_rejected(pdf):
    result = validate_pdf(compilation(pdf), OptimizedResume())
    assert not result.success and result.pdf_bytes is None and result.error


def test_missing_contact_or_bullet_rejects_one_page_pdf():
    resume = OptimizedResume(contact=ContactInfo(name="Alex Rivera", email="alex@example.com"))
    result = validate_pdf(compilation(make_pdf()), resume)
    assert not result.success and result.page_count == 1 and result.pdf_bytes is None


def test_unicode_contact_is_not_silently_ignored_by_completeness_checks():
    resume = OptimizedResume(contact=ContactInfo(name="李明"))
    result = validate_pdf(compilation(make_pdf()), resume)
    assert not result.success and result.pdf_bytes is None


def test_compiler_overflow_rejects_pdf_even_with_one_page():
    comp = compilation(make_pdf())
    comp.warnings.append("Text overflows the printable layout")
    assert not validate_pdf(comp, OptimizedResume()).success


def test_escape_does_not_escape_inserted_commands_or_use_html_entities():
    assert latex_escape("A&B <C> \\ {x}") == r"A\&B <C> \textbackslash{} \{x\}"
    source = render_resume_latex(OptimizedResume(summary="Owned C++ & Python <services>"))
    assert r"C++ \& Python <services>" in source
    assert "&amp;" not in source and not validate_latex_source(source)


@pytest.mark.parametrize("injection", [
    r"\input{/etc/passwd}", r"\include{private}", r"\write18{whoami}",
    r"\csname input\endcsname{private}", r"\catcode`\@=0", r"\usepackage{shellesc}",
    r"\special{pdf:literal test}", r"\newcommand{\attack}{x}", "^^5cinput{private}",
    r"\vspace{-100em}", r"\\[-100pt]", r"\renewcommand{\section}{x}", r"\small tiny resume",
    r"\titlespacing*{\section}{0pt}{-100em}{0em}",
])
def test_untrusted_tex_commands_are_blocked_before_process_execution(monkeypatch, injection):
    source = render_resume_latex(OptimizedResume(summary="Resume facts."))
    source = source.replace(r"\end{document}", injection + "\n" + r"\end{document}")
    monkeypatch.setattr(compiler, "find_compiler", lambda: pytest.fail("Compiler lookup must follow source validation"))
    result = compile_latex(source)
    assert not result.success and result.pdf_bytes is None and result.error


def test_secret_environment_is_not_passed_to_compiler(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_API_KEY", "other-secret")
    env = _restricted_env()
    assert "ANTHROPIC_API_KEY" not in env and "LLM_API_KEY" not in env
    assert env["openin_any"] == "p" and env["TECTONIC_UNTRUSTED_MODE"] == "1"


def test_tectonic_uses_untrusted_mode_and_private_temp_directory(monkeypatch):
    seen = {}
    monkeypatch.setattr(compiler, "find_compiler", lambda: "tectonic.exe")
    def run(cmd, **kwargs):
        seen.update(kwargs)
        seen["cmd"] = cmd
        (Path(kwargs["cwd"]) / "resume.pdf").write_bytes(make_pdf())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(compiler.subprocess, "run", run)
    result = compile_latex(render_resume_latex(OptimizedResume(summary="Resume facts.")))
    assert result.success and "--untrusted" in seen["cmd"]
    assert not Path(seen["cwd"]).exists()


def _long_resume():
    return OptimizedResume(contact=ContactInfo(name="Alex Rivera"), experience=[
        OptimizedRole(title="Software Engineer", employer="Acme", bullets=[
            OptimizedBullet(text=f"Built reliable service number {i} with a complete factual ending.")
            for i in range(5)
        ])
    ])


def test_tightening_removes_whole_bullets_and_preserves_source():
    resume = _long_resume()
    fitted = tighten_content(resume)
    assert len(fitted.experience[0].bullets) == 4
    assert len(resume.experience[0].bullets) == 5
    assert all(b.text.endswith("complete factual ending.") for b in fitted.experience[0].bullets)


def test_fitter_retries_spacing_then_removes_content_and_accepts_only_one_page(monkeypatch):
    calls = []
    def fake_compile(source):
        calls.append(source)
        return compilation(make_pdf(pages=1 if source.count(r"\item") <= 4 else 2))
    monkeypatch.setattr(compiler, "compile_latex", fake_compile)
    # The synthetic PDF contains no generated body; use only real page validation.
    import app.latex.pdf_validate as validation
    real_validate = validation.validate_pdf
    monkeypatch.setattr(validation, "validate_pdf", lambda comp, resume: real_validate(comp, OptimizedResume()))
    fitted, source, result, decisions = fit_resume_to_one_page(_long_resume())
    assert len(calls) == 3 and len(fitted.experience[0].bullets) == 4
    assert result.success and result.page_count == 1 and result.layout_verified
    assert "margin=0.5in" in source and "10pt" in source and decisions


def test_fitter_exhaustion_never_releases_multiple_pages(monkeypatch):
    monkeypatch.setattr(compiler, "compile_latex", lambda source: compilation(make_pdf(pages=2)))
    fitted, source, result, decisions = fit_resume_to_one_page(_long_resume())
    assert not result.success and result.pdf_bytes is None and result.page_count == 2
    assert len(fitted.experience[0].bullets) >= 1 and source and result.error
