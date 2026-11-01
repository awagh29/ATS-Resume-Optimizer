"""Browser workflow, private artifact downloads, and readiness diagnostics."""

from __future__ import annotations

import io
import secrets
import sqlite3
from pathlib import Path

from flask import (
    Blueprint, abort, current_app, flash, jsonify, redirect, render_template,
    request, send_file, session, url_for,
)

from app.artifacts import ArtifactStore
from app.llm.adapter import LLMError, get_llm_client
from app.llm.demo import DEMO_NOTICE
from app.pipeline.orchestrator import run_optimization

bp = Blueprint("main", __name__)


def _store() -> ArtifactStore:
    return current_app.extensions["resume_artifacts"]


def _knowledge_store():
    return current_app.extensions.get("knowledge_store")


def _knowledge_unavailable():
    flash("Your local library is unavailable. Check that the app can access its data folder and try again.", "error")


def _library_counts() -> dict:
    store = _knowledge_store()
    if store is None:
        return {"knowledge_count": 0, "history_count": 0, "library_available": False}
    try:
        return {"knowledge_count": len(store.list_entries()),
                "history_count": len(store.list_runs()), "library_available": True}
    except (OSError, sqlite3.Error):
        return {"knowledge_count": 0, "history_count": 0, "library_available": False}


def _session_key() -> str:
    if "sid" not in session:
        session["sid"] = secrets.token_urlsafe(32)
    return session["sid"]


def csrf_token() -> str:
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


@bp.before_app_request
def protect_forms():
    # Drop legacy client-side document fields after upgrades.
    session.pop("last_latex", None)
    session.pop("last_page_size", None)
    if request.method == "POST":
        expected = session.get("csrf_token", "")
        received = request.form.get("csrf_token", "")
        if (not expected or not received
                or not secrets.compare_digest(expected.encode("utf-8"), received.encode("utf-8"))):
            abort(400, description="Your form expired. Open a new optimization and try again.")


@bp.after_app_request
def private_responses(response):
    if request.endpoint != "static":
        response.headers["Cache-Control"] = "no-store, private"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


@bp.app_context_processor
def template_helpers():
    return {"csrf_token": csrf_token}


def _valid_pdf(compilation) -> bool:
    return bool(compilation.success and compilation.layout_verified
                and compilation.page_count == 1 and compilation.pdf_bytes)


@bp.route("/")
def index():
    return _render_index()


def _render_index(input_values=None):
    from app.latex.compiler import find_compiler

    sample_resume = sample_jd = ""
    try:
        base = Path(current_app.root_path).parent / "samples"
        sample_resume = (base / "sample_resume.tex").read_text(encoding="utf-8")
        sample_jd = (base / "sample_job_description.txt").read_text(encoding="utf-8")
    except OSError:
        pass
    client = get_llm_client()
    return render_template(
        "index.html", demo_mode=not client.configured,
        demo_notice=DEMO_NOTICE if not client.configured else "",
        claude_model=client.model, compiler_available=bool(find_compiler()),
        sample_resume=sample_resume, sample_jd=sample_jd,
        max_resume=current_app.config["MAX_RESUME_CHARS"],
        max_jd=current_app.config["MAX_JD_CHARS"],
        max_supp=current_app.config["MAX_SUPPLEMENTAL_CHARS"],
        input_values=input_values or {},
        **_library_counts(),
    )


def _retry_index(resume_latex=None):
    values = request.form.to_dict()
    if resume_latex is not None:
        values["resume_latex"] = resume_latex
    return _render_index(values)


@bp.route("/health")
def health():
    from app.latex.compiler import find_compiler

    claude = get_llm_client().configured
    compiler = bool(find_compiler())
    return jsonify(status="ready" if claude and compiler else "setup_required",
                   claude_configured=claude, compiler_available=compiler,
                   one_page_required=True)


@bp.route("/optimize", methods=["POST"])
def optimize():
    sid = _session_key()
    _store().delete(sid)
    resume_latex = (request.form.get("resume_latex") or "").strip()
    max_resume = current_app.config["MAX_RESUME_CHARS"]
    upload = request.files.get("resume_file")
    if upload and upload.filename:
        if Path(upload.filename).suffix.lower() != ".tex":
            flash("Upload a .tex file containing UTF-8 LaTeX source.", "error")
            return _retry_index()
        try:
            raw = upload.read(max_resume * 4 + 1)
            if len(raw) > max_resume * 4:
                flash("Uploaded .tex file is too large.", "error")
                return _retry_index()
            resume_latex = raw.decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            flash("Save the .tex file as UTF-8 text and upload it again.", "error")
            return _retry_index()
        except OSError:
            flash("Could not read uploaded file.", "error")
            return _retry_index()

    jd = (request.form.get("job_description") or "").strip()
    company = (request.form.get("company") or "").strip()[:200]
    role_title = (request.form.get("role_title") or "").strip()[:200]
    supplemental = (request.form.get("supplemental") or "").strip()
    page_size = (request.form.get("page_size") or "letter").strip().lower()
    errors = []
    if not resume_latex:
        errors.append("Resume LaTeX is required (paste or upload a .tex file).")
    if not jd:
        errors.append("Job description is required.")
    if len(resume_latex) > max_resume:
        errors.append(f"Resume exceeds {max_resume} characters.")
    if len(jd) > current_app.config["MAX_JD_CHARS"]:
        errors.append("Job description exceeds the allowed length.")
    if len(supplemental) > current_app.config["MAX_SUPPLEMENTAL_CHARS"]:
        errors.append("Supplemental facts exceed the allowed length.")
    if page_size not in {"letter", "a4"}:
        errors.append("Choose US Letter or A4 as the page size.")
    if errors:
        for error in errors:
            flash(error, "error")
        return _retry_index(resume_latex)

    try:
        result = run_optimization(
            resume_latex, jd, company=company, role_title=role_title,
            supplemental_text=supplemental, page_size=page_size,
            llm=get_llm_client(),
            knowledge_store=_knowledge_store(),
            use_knowledge=request.form.get("use_knowledge", "1") != "0",
        )
    except LLMError as exc:
        flash(str(exc), "error")
        return _retry_index(resume_latex)
    except Exception as exc:  # Do not log exception messages or source document bodies.
        current_app.logger.error("Optimization failed (%s)", type(exc).__name__)
        flash("Optimization could not complete. Check your provider and compiler configuration, then try again.", "error")
        return _retry_index(resume_latex)

    has_pdf = _valid_pdf(result.compilation)
    stored = _store().put(sid, latex=result.latex_source,
                          pdf=result.compilation.pdf_bytes if has_pdf else None,
                          page_size=page_size)
    if not stored:
        flash("The result exceeded temporary storage limits. LaTeX is still available below.", "error")
        has_pdf = False
    return render_template(
        "results.html", result=result.to_template_dict(),
        latex_source=result.latex_source, has_pdf=has_pdf,
        demo_mode=result.demo_mode, demo_notice=result.demo_notice,
        page_size=page_size,
    )


@bp.route("/knowledge", methods=["GET", "POST"])
def knowledge():
    store = _knowledge_store()
    values = request.form.to_dict() if request.method == "POST" else {}
    if request.method == "POST" and store is not None:
        title = (request.form.get("title") or "").strip()
        content = (request.form.get("content") or "").strip()
        category = (request.form.get("category") or "accomplishment").strip()
        if not title or not content:
            flash("Add a title and the confirmed facts you want to save.", "error")
        elif len(title) > 200 or len(content) > 10000:
            flash("Keep the title within 200 characters and the facts within 10,000 characters.", "error")
        elif category not in {"accomplishment", "skill", "project", "metric", "responsibility"}:
            flash("Choose one of the listed categories.", "error")
        elif request.form.get("confirmed") != "1":
            flash("Confirm that these facts describe your own verified experience before saving.", "error")
        else:
            try:
                store.add_entry(title, content, category=category)
            except ValueError as exc:
                flash(str(exc), "error")
            except (OSError, sqlite3.Error):
                _knowledge_unavailable()
            else:
                flash("Knowledge saved. Relevant facts will be considered in future optimizations.", "success")
                return redirect(url_for("main.knowledge"))
    entries = []
    available = store is not None
    if available:
        try:
            entries = store.list_entries()
        except (OSError, sqlite3.Error):
            available = False
    if not available:
        _knowledge_unavailable()
    return render_template("knowledge.html", entries=entries, input_values=values,
                           library_available=available)


@bp.route("/knowledge/<entry_id>/delete", methods=["POST"])
def delete_knowledge(entry_id):
    store = _knowledge_store()
    try:
        removed = store.delete_entry(entry_id) if store else False
    except (OSError, sqlite3.Error):
        _knowledge_unavailable()
    else:
        flash("Knowledge removed from future optimizations. Existing history retains its source snapshot."
              if removed else "This knowledge entry is no longer available.", "success" if removed else "error")
    return redirect(url_for("main.knowledge"))


@bp.route("/history")
def history():
    store = _knowledge_store()
    runs = []
    available = store is not None
    if available:
        try:
            runs = store.list_runs(limit=50)
        except (OSError, sqlite3.Error):
            available = False
    if not available:
        _knowledge_unavailable()
    return render_template("history.html", runs=runs, library_available=available)


@bp.route("/history/<run_id>")
def history_result(run_id):
    store = _knowledge_store()
    try:
        run = store.get_run(run_id) if store else None
    except (OSError, sqlite3.Error):
        _knowledge_unavailable()
        return redirect(url_for("main.history"))
    if run is None:
        abort(404, description="This saved optimization is no longer available.")
    result = run["result"]
    latex = result.get("latex_source", "")
    page_size = result.get("page_size", "letter")
    # Saved history contains editable output, never an unvalidated PDF. Restoring
    # it uses no model calls; the user can recompile locally when needed.
    sid = _session_key()
    _store().delete(sid)
    _store().put(sid, latex=latex, page_size=page_size)
    return render_template("results.html", result=result, latex_source=latex,
                           has_pdf=False, demo_mode=result.get("demo_mode", False),
                           demo_notice=result.get("demo_notice", ""), page_size=page_size,
                           saved_run=run)


@bp.route("/history/<run_id>/delete", methods=["POST"])
def delete_history(run_id):
    store = _knowledge_store()
    try:
        removed = store.delete_run(run_id) if store else False
    except (OSError, sqlite3.Error):
        _knowledge_unavailable()
    else:
        flash("Saved optimization deleted." if removed else "This saved optimization is no longer available.",
              "success" if removed else "error")
    return redirect(url_for("main.history"))


@bp.route("/download/tex", methods=["POST"])
def download_tex():
    artifact = _store().get(session.get("sid", ""))
    latex = request.form.get("latex_source")
    if latex is None:
        latex = artifact.latex if artifact else ""
    if not latex.strip():
        flash("No LaTeX available. Generate a resume first.", "error")
        return redirect(url_for("main.index"))
    if len(latex) > current_app.config["MAX_RESUME_CHARS"]:
        abort(413)
    return send_file(io.BytesIO(latex.encode("utf-8")), as_attachment=True,
                     download_name="optimized_resume.tex", mimetype="application/x-tex")


def _pdf_response(*, attachment: bool):
    artifact = _store().get(session.get("sid", ""))
    if not artifact or not artifact.pdf:
        if not attachment:
            abort(404, description="This PDF has expired or has not passed the one-page checks.")
        flash("No verified one-page PDF available. Generate or recompile a resume first; temporary results expire after one hour.", "error")
        return redirect(url_for("main.index"))
    return send_file(io.BytesIO(artifact.pdf), as_attachment=attachment,
                     download_name="optimized_resume.pdf", mimetype="application/pdf")


@bp.route("/download/pdf")
def download_pdf():
    return _pdf_response(attachment=True)


@bp.route("/preview/pdf")
def preview_pdf():
    return _pdf_response(attachment=False)


@bp.route("/preview/image")
def preview_image():
    """Render a session-private preview without requiring a browser PDF viewer."""
    artifact = _store().get(session.get("sid", ""))
    if not artifact or not artifact.pdf:
        abort(404, description="No verified resume preview is available.")
    try:
        from app.latex.preview import render_pdf_preview

        png = render_pdf_preview(artifact.pdf)
    except Exception as exc:
        current_app.logger.warning("Resume preview unavailable (%s)", type(exc).__name__)
        abort(404, description="The preview is unavailable. You can still download your PDF.")
    return send_file(io.BytesIO(png), mimetype="image/png",
                     download_name="resume_preview.png")


@bp.route("/recompile", methods=["POST"])
def recompile():
    """Only publish edited output after it passes the same strict PDF checks."""
    from app.latex.compiler import compile_latex
    from app.latex.pdf_validate import validate_pdf
    from app.models.schemas import OptimizedResume

    sid = _session_key()
    artifact = _store().get(sid)
    _store().delete(sid)
    latex = request.form.get("latex_source") or ""
    if not latex.strip():
        flash("LaTeX source is empty.", "error")
        return redirect(url_for("main.index"))
    if len(latex) > current_app.config["MAX_RESUME_CHARS"]:
        flash("LaTeX source is too large.", "error")
        return redirect(url_for("main.index"))
    compilation = compile_latex(latex)
    if compilation.success and compilation.pdf_bytes:
        compilation = validate_pdf(compilation, OptimizedResume())
    has_pdf = _valid_pdf(compilation)
    stored = _store().put(sid, latex=latex,
                          pdf=compilation.pdf_bytes if has_pdf else None,
                          page_size=artifact.page_size if artifact else "letter")
    has_pdf = has_pdf and stored
    error = "" if has_pdf else (compilation.error or "The PDF did not pass the one-page content and layout checks.")
    return render_template(
        "recompile.html", latex_source=latex, has_pdf=has_pdf,
        page_count=compilation.page_count, warnings=compilation.warnings,
        layout_verified=has_pdf, error=error,
    )


@bp.app_errorhandler(413)
def too_large(_exc):
    _store().delete(session.get("sid", ""))
    return render_template("error.html", message="The upload or form is too large. Reduce its size and try again."), 413


@bp.app_errorhandler(400)
def bad_request(exc):
    return render_template("error.html", message=exc.description), 400
