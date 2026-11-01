"""Local library forms, saved output recovery, and no-request history browsing."""

import sqlite3

import pytest

from app import create_app
from app.config import Config
from app.knowledge import KnowledgeStore
from app.models.schemas import OptimizationResult


class LibraryTestConfig(Config):
    TESTING = True
    SECRET_KEY = "local-library-tests"
    ANTHROPIC_API_KEY = ""
    KNOWLEDGE_DB_PATH = ":memory:"


@pytest.fixture
def app():
    return create_app(LibraryTestConfig)


@pytest.fixture
def client(app):
    return app.test_client()


def token(client):
    assert client.get("/").status_code == 200
    with client.session_transaction() as session:
        return session["csrf_token"]


def test_library_forms_require_csrf_and_confirmed_source(client, app):
    store = app.extensions["knowledge_store"]
    assert client.post("/knowledge", data={"title": "New fact"}).status_code == 400
    csrf = token(client)
    response = client.post("/knowledge", data={
        "csrf_token": csrf, "title": "BrightPath payments", "category": "metric",
        "content": "At BrightPath Labs, reduced p95 latency to 220ms.",
    })
    assert b"Confirm that these facts" in response.data
    assert store.list_entries() == []
    response = client.post("/knowledge", data={
        "csrf_token": csrf, "title": "BrightPath payments", "category": "metric",
        "content": "At BrightPath Labs, reduced p95 latency to 220ms.", "confirmed": "1",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"Knowledge saved" in response.data
    assert len(store.list_entries()) == 1
    entry_id = store.list_entries()[0]["id"]
    assert client.get(f"/knowledge/{entry_id}/delete").status_code == 405
    assert client.post(f"/knowledge/{entry_id}/delete").status_code == 400
    assert len(store.list_entries()) == 1
    response = client.post(f"/knowledge/{entry_id}/delete", data={"csrf_token": csrf},
                           follow_redirects=True)
    assert response.status_code == 200
    assert store.list_entries() == []


def test_library_entries_escape_html_and_reject_bad_category(client, app):
    csrf = token(client)
    app.extensions["knowledge_store"].add_entry("<script>title</script>", "<script>source</script>")
    response = client.get("/knowledge")
    assert b"&lt;script&gt;source&lt;/script&gt;" in response.data
    assert b"<script>source</script>" not in response.data
    response = client.post("/knowledge", data={
        "csrf_token": csrf, "title": "Invalid", "content": "Fact", "category": "invented",
        "confirmed": "1",
    })
    assert b"Choose one of the listed categories" in response.data
    assert len(app.extensions["knowledge_store"].list_entries()) == 1


def test_history_restores_editable_output_without_model_requests(client, app, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Viewing saved history must not invoke optimization or make a model request")

    monkeypatch.setattr("app.routes.run_optimization", forbidden)
    monkeypatch.setattr("app.llm.adapter.LLMClient.chat_json", forbidden)
    result = OptimizationResult(latex_source="SAVED_RESUME_OUTPUT", page_size="a4",
                                token_usage={"api_calls": 2, "input_tokens": 40,
                                             "output_tokens": 20, "local_cache_hits": 1},
                                knowledge_used=[{"id": "source-id", "title": "Confirmed API metrics",
                                                 "category": "metric", "content": "Reduced latency by 47% <verified>",
                                                 "fact_ids": ["supplemental-1"]}])
    result.job_analysis.role_title = "Backend Engineer"
    run_id = app.extensions["knowledge_store"].save_run(
        result.to_template_dict(), job_description="Build Python APIs", company="BrightPath",
        role_title="Backend Engineer")
    assert b"Backend Engineer" in client.get("/history").data
    csrf = token(client)
    with client.session_transaction() as session:
        session["sid"] = "prior-result"
    app.extensions["resume_artifacts"].put("prior-result", latex="old", pdf=b"%PDF-stale")
    response = client.get(f"/history/{run_id}")
    assert response.status_code == 200
    assert b"Saved optimization" in response.data
    assert b"SAVED_RESUME_OUTPUT" in response.data
    assert b"A4" in response.data
    assert b"Locally reused responses" in response.data
    assert b"Confirmed API metrics" in response.data
    assert b"Reduced latency by 47% &lt;verified&gt;" in response.data
    assert b"/preview/image" not in response.data
    assert client.get("/preview/pdf").status_code == 404
    download = client.post("/download/tex", data={"csrf_token": csrf})
    assert download.data == b"SAVED_RESUME_OUTPUT"
    artifact = app.extensions["resume_artifacts"].get("prior-result")
    assert artifact.page_size == "a4"
    assert artifact.pdf is None


def test_history_delete_keeps_confirmed_knowledge(client, app):
    store = app.extensions["knowledge_store"]
    store.add_entry("Python", "Skill: Python")
    run_id = store.save_run(OptimizationResult().to_template_dict(), job_description="Python")
    assert client.post(f"/history/{run_id}/delete").status_code == 400
    assert store.get_run(run_id)
    response = client.post(f"/history/{run_id}/delete", data={"csrf_token": token(client)},
                           follow_redirects=True)
    assert b"Saved optimization deleted" in response.data
    assert store.get_run(run_id) is None
    assert len(store.list_entries()) == 1
    assert client.get(f"/history/{run_id}").status_code == 404


def test_optimize_passes_knowledge_preference(client, app, monkeypatch):
    received = []

    def optimize(*args, **kwargs):
        received.append(kwargs)
        return OptimizationResult(latex_source="output")

    monkeypatch.setattr("app.routes.run_optimization", optimize)
    csrf = token(client)
    for value in ("0", "1"):
        response = client.post("/optimize", data={
            "csrf_token": csrf, "resume_latex": "source", "job_description": "job",
            "use_knowledge": value,
        })
        assert response.status_code == 200
    assert received[0]["use_knowledge"] is False
    assert received[1]["use_knowledge"] is True
    assert received[0]["knowledge_store"] is app.extensions["knowledge_store"]


def test_unavailable_store_shows_recovery_message(client, app, monkeypatch):
    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("Private storage details")

    monkeypatch.setattr(app.extensions["knowledge_store"], "list_entries", unavailable)
    response = client.get("/knowledge")
    assert response.status_code == 200
    assert b"Your local library is unavailable" in response.data
    assert b"Private storage details" not in response.data
    assert b"disabled" in response.data
