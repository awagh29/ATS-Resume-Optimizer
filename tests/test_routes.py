"""Browser contracts: private artifacts, strict PDF gates, and input protection."""

from io import BytesIO

import pytest

from app import create_app
from app.artifacts import ArtifactStore
from app.config import Config
from app.llm.adapter import LLMError
from app.models.schemas import CompilationResult, OptimizationResult


class TestConfig(Config):
    TESTING = True
    SECRET_KEY = "route-test-secret"
    ANTHROPIC_API_KEY = ""
    LLM_API_KEY = ""


@pytest.fixture
def app():
    return create_app(TestConfig)


@pytest.fixture
def client(app):
    return app.test_client()


def _token(client):
    response = client.get("/")
    assert response.status_code == 200
    with client.session_transaction() as session:
        return session["csrf_token"]


def _result(*, valid=True):
    return OptimizationResult(
        latex_source=r"\documentclass{article}\begin{document}PRIVATE_RESUME_CONTENT\end{document}",
        compilation=CompilationResult(
            success=True, compiler_available=True, pdf_bytes=b"%PDF-private-test",
            page_count=1 if valid else 2, layout_verified=valid,
        ),
    )


def _optimize(client, monkeypatch, *, valid=True):
    monkeypatch.setattr("app.routes.run_optimization", lambda *args, **kwargs: _result(valid=valid))
    return client.post("/optimize", data={
        "csrf_token": _token(client), "resume_latex": "Source resume",
        "job_description": "Source job description", "page_size": "letter",
    })


def test_post_requires_session_csrf(client, monkeypatch):
    monkeypatch.setattr("app.routes.run_optimization", lambda *a, **kw: pytest.fail("CSRF bypass"))
    assert client.post("/optimize", data={"resume_latex": "source"}).status_code == 400
    _token(client)
    assert client.post("/download/tex", data={"csrf_token": "invalid"}).status_code == 400
    assert client.post("/download/tex", data={"csrf_token": "invalid\u00e9"}).status_code == 400


def test_optimized_artifacts_stay_out_of_cookie(client, app, monkeypatch):
    response = _optimize(client, monkeypatch)
    assert response.status_code == 200
    assert b"/preview/image" in response.data
    assert b"data:application/pdf" not in response.data
    assert response.data.count(b"<h1>") == 1
    assert b"Your one-page resume is ready" in response.data
    assert response.data.index(b"Download PDF") < response.data.index(b"/preview/image")
    assert response.data.index(b"/preview/image") < response.data.index(b"Match estimate:")
    with client.session_transaction() as session:
        assert set(session) <= {"sid", "csrf_token"}
        assert "PRIVATE_RESUME_CONTENT" not in str(dict(session))
    assert client.get("/download/pdf").data == b"%PDF-private-test"
    preview = client.get("/preview/pdf")
    assert preview.status_code == 200
    assert "no-store" in preview.headers["Cache-Control"]
    assert "inline" in preview.headers["Content-Disposition"]
    assert app.test_client().get("/preview/pdf").status_code == 404
    tex = client.post("/download/tex", data={"csrf_token": _token(client)})
    assert b"PRIVATE_RESUME_CONTENT" in tex.data


def test_new_unverified_result_clears_previous_pdf(client, monkeypatch):
    _optimize(client, monkeypatch)
    assert client.get("/download/pdf").status_code == 200
    response = _optimize(client, monkeypatch, valid=False)
    assert response.status_code == 200
    assert b"/preview/image" not in response.data
    assert b"Review your resume output" in response.data
    assert b'id="latex-editor" open' in response.data
    assert client.get("/preview/pdf").status_code == 404
    assert client.get("/download/pdf").status_code == 302


def test_provider_error_preserves_input_and_clears_old_pdf(client, monkeypatch):
    _optimize(client, monkeypatch)

    def fail_provider(*args, **kwargs):
        raise LLMError("Claude key is invalid. Update ANTHROPIC_API_KEY.")

    monkeypatch.setattr("app.routes.run_optimization", fail_provider)
    response = client.post("/optimize", data={
        "csrf_token": _token(client), "resume_latex": "Preserve my resume",
        "job_description": "Preserve my target role", "page_size": "a4",
    })
    assert response.status_code == 200
    assert b"Claude key is invalid" in response.data
    assert b"Preserve my resume" in response.data
    assert b"Preserve my target role" in response.data
    assert b'value="a4" selected' in response.data
    assert client.get("/preview/pdf").status_code == 404


@pytest.mark.parametrize("compile_ok", [False, True])
def test_failed_recompile_never_serves_stale_pdf(client, monkeypatch, compile_ok):
    _optimize(client, monkeypatch)
    comp = CompilationResult(success=compile_ok, compiler_available=True,
                             pdf_bytes=b"rejected" if compile_ok else None,
                             error="failed" if not compile_ok else "")
    monkeypatch.setattr("app.latex.compiler.compile_latex", lambda *a, **kw: comp)
    monkeypatch.setattr("app.latex.pdf_validate.validate_pdf", lambda value, resume:
                        CompilationResult(success=False, error="Exactly one page required.", page_count=2))
    response = client.post("/recompile", data={
        "csrf_token": _token(client), "latex_source": "edited source",
    })
    assert response.status_code == 200
    assert b"/preview/image" not in response.data
    assert client.get("/preview/pdf").status_code == 404


def test_successful_recompile_publishes_only_validated_pdf(client, monkeypatch):
    token = _token(client)
    comp = CompilationResult(success=True, compiler_available=True,
                             pdf_bytes=b"%PDF-edited-resume")
    monkeypatch.setattr("app.latex.compiler.compile_latex", lambda *a, **kw: comp)

    def accept(value, resume):
        value.layout_verified = True
        value.page_count = 1
        return value

    monkeypatch.setattr("app.latex.pdf_validate.validate_pdf", accept)
    response = client.post("/recompile", data={
        "csrf_token": token, "latex_source": "edited source",
    })
    assert response.status_code == 200
    assert b"Verified as exactly one page" in response.data
    assert client.get("/preview/pdf").data == b"%PDF-edited-resume"


def test_rendered_preview_is_private_and_uses_current_pdf(client, app, monkeypatch):
    png = b"\x89PNG\r\n\x1a\nrendered-test-preview"
    received = []

    def render(pdf):
        received.append(pdf)
        return png

    monkeypatch.setattr("app.latex.preview.render_pdf_preview", render)
    assert client.get("/preview/image").status_code == 404
    assert received == []
    _optimize(client, monkeypatch)
    image = client.get("/preview/image")
    assert image.status_code == 200
    assert image.mimetype == "image/png"
    assert image.data == png
    assert "no-store" in image.headers["Cache-Control"]
    assert received == [b"%PDF-private-test"]
    assert app.test_client().get("/preview/image").status_code == 404
    _optimize(client, monkeypatch, valid=False)
    assert client.get("/preview/image").status_code == 404
    assert len(received) == 1


def test_preview_render_failure_preserves_pdf_download(client, monkeypatch):
    _optimize(client, monkeypatch)

    def fail_render(_pdf):
        raise ValueError("PRIVATE_DETAILS_MUST_NOT_LEAK")

    monkeypatch.setattr("app.latex.preview.render_pdf_preview", fail_render)
    response = client.get("/preview/image")
    assert response.status_code == 404
    assert b"PRIVATE_DETAILS_MUST_NOT_LEAK" not in response.data
    assert "no-store" in response.headers["Cache-Control"]
    assert client.get("/download/pdf").data == b"%PDF-private-test"


@pytest.mark.parametrize("filename,raw,expected", [
    ("resume.pdf", b"pdf", b"Upload a .tex"),
    ("resume.tex", b"\xff\xfe", b"Save the .tex file as UTF-8"),
])
def test_invalid_upload_rejected(client, monkeypatch, filename, raw, expected):
    monkeypatch.setattr("app.routes.run_optimization", lambda *a, **kw: pytest.fail("invalid upload"))
    response = client.post("/optimize", data={
        "csrf_token": _token(client), "resume_file": (BytesIO(raw), filename),
        "job_description": "role",
    }, follow_redirects=True)
    assert expected in response.data


def test_request_size_limit_and_custom_config(client, app, monkeypatch):
    app.config["MAX_RESUME_CHARS"] = 5
    token = _token(client)
    monkeypatch.setattr("app.routes.run_optimization", lambda *a, **kw: pytest.fail("oversized input"))
    response = client.post("/optimize", data={
        "csrf_token": token, "resume_latex": "sixsix", "job_description": "jd",
    }, follow_redirects=True)
    assert b"Resume exceeds 5 characters" in response.data
    app.config["MAX_CONTENT_LENGTH"] = 100
    assert client.post("/optimize", data={"csrf_token": token, "resume_latex": "x" * 1000}).status_code == 413


def test_health_has_no_credentials(client, monkeypatch):
    monkeypatch.setattr("app.latex.compiler.find_compiler", lambda: None)
    response = client.get("/health")
    assert response.json == {"status": "setup_required", "claude_configured": False,
                             "compiler_available": False, "one_page_required": True}
    assert b"route-test-secret" not in response.data


def test_artifact_store_expires_and_bounds_memory(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("app.artifacts.time.monotonic", lambda: now[0])
    store = ArtifactStore(ttl=5, max_entries=2, max_bytes=8)
    assert store.put("one", latex="1111")
    assert store.put("two", latex="2222")
    assert store.get("one")
    assert store.put("three", latex="3333")
    assert store.get("two") is None  # Least recently read entry is evicted.
    assert not store.put("large", latex="x" * 9)
    now[0] = 16
    assert store.get("one") is None
    assert store.get("three") is None


def test_artifact_stores_are_per_app(app):
    other = create_app(TestConfig)
    app.extensions["resume_artifacts"].put("sid", latex="private")
    assert other.extensions["resume_artifacts"].get("sid") is None
