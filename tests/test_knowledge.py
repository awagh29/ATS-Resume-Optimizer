"""Durable knowledge, bounded retrieval, history isolation, and local caching."""

from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from app.knowledge import KnowledgeStore


@pytest.fixture
def store():
    database = KnowledgeStore(":memory:")
    yield database
    database.close()


def test_knowledge_persists_after_reopening(tmp_path):
    path = tmp_path / "private" / "knowledge.sqlite3"
    first = KnowledgeStore(path)
    entry = first.add_entry("Warehouse", "Built Python pipelines for warehouse data.", "project")
    first.close()
    reopened = KnowledgeStore(path)
    try:
        assert reopened.list_entries() == [entry]
        assert entry["created_at"].endswith("Z")
        assert reopened._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        reopened.close()


def test_exact_duplicate_notes_reuse_original_provenance(store):
    original = store.add_entry("Python", "Built a Python service.", "skill")
    duplicate = store.add_entry(" Python ", "Built a Python service. ", "skill")
    assert duplicate == original
    assert len(store.list_entries()) == 1


def test_candidate_profiles_are_isolated_across_all_store_operations(store):
    alice = store.add_entry("Python", "Built Python services.", profile_id="alice")
    bob = store.add_entry("SQL", "Maintained SQL databases.", profile_id="bob")
    assert store.list_entries() == []
    assert store.list_entries("alice") == [alice]
    assert not store.delete_entry(alice["id"], "bob")
    assert store.retrieve("Python", "bob") == []
    assert store.retrieve("SQL", "bob")[0]["id"] == bob["id"]
    run = store.save_run({"latex_source": "private"}, job_description="Python", profile_id="alice")
    assert store.get_run(run, "bob") is None
    assert store.list_runs("bob") == []
    assert not store.delete_run(run, "bob")
    assert store.delete_run(run, "alice")


def test_retrieval_ranks_relevant_notes_and_retains_sources(store):
    store.add_entry("Sales", "Won a customer renewal.")
    store.add_entry("General development", "Wrote Python scripts and maintained Excel reports.")
    focused = store.add_entry("Python data pipelines", "Built Python ETL pipelines using SQL and Airflow.")
    hits = store.retrieve("Python ETL data pipelines using Airflow and SQL")
    assert hits[0]["id"] == focused["id"]
    assert all(hit["title"] != "Sales" for hit in hits)
    assert hits[0]["source"] == f"knowledge:{focused['id']}"
    assert hits[0]["truncated"] is False
    assert store.retrieve("the and required experience skills") == []


def test_retrieval_budget_accounts_for_escaping_unicode_and_metadata(store):
    oversized = store.add_entry("Python \"service\"", 'Python résumé \\ "service".\n' * 300)
    note = store.add_entry("Python", 'Built Python résumé \\ "service".')
    hits = store.retrieve("Python service", max_chars=700)
    assert len(json.dumps(hits, ensure_ascii=False, separators=(",", ":"))) <= 700
    assert hits[0]["id"] == note["id"]
    assert hits[0]["truncated"] is False
    assert hits[0]["content"] == note["content"]
    assert all(hit["id"] != oversized["id"] for hit in hits)
    assert store.retrieve("Python", max_chars=10) == []
    assert store.retrieve("Python", limit=0) == []


def test_generated_resume_history_never_becomes_knowledge(store):
    store.save_run({"optimized_resume": {"summary": "Kubernetes expert"}}, job_description="Kubernetes")
    store.set_cached_response("example", {"skills": ["Kubernetes"]})
    assert store.retrieve("Kubernetes") == []
    assert store.list_entries() == []


def test_history_keeps_structured_output_and_excludes_pdf_bytes(store):
    result = {
        "latex_source": "\\documentclass{article}",
        "page_size": "a4",
        "original_assessment": {"score": {"total": 61.2}},
        "optimized_assessment": {"score": {"total": 96.0}},
        "target_reached": True,
        "demo_mode": True,
        "compilation": {"pdf_bytes": b"binary-pdf", "success": True},
    }
    run_id = store.save_run(result, job_description="Engineer role", company="Acme",
                            role_title="Engineer", resume_hash="source-hash")
    record = store.get_run(run_id)
    assert record["job_description"] == "Engineer role"
    assert record["resume_hash"] == "source-hash"
    assert record["result"]["run_id"] == run_id
    assert record["result"]["latex_source"] == result["latex_source"]
    assert record["result"]["page_size"] == "a4"
    assert "pdf_bytes" not in record["result"]["compilation"]
    # The caller's result is not mutated.
    assert result["compilation"]["pdf_bytes"] == b"binary-pdf"
    summary = store.list_runs()[0]
    assert summary["optimized_score"] == 96.0
    assert summary["original_score"] == 61.2
    assert summary["company"] == "Acme"
    assert summary["target_reached"] and summary["demo_mode"]
    assert "result" not in summary and "job_description" not in summary


def test_history_is_newest_first_and_can_be_deleted(store):
    first = store.save_run({}, job_description="first")
    second = store.save_run({}, job_description="second")
    assert [run["id"] for run in store.list_runs(limit=1)] == [second]
    assert store.delete_run(first)
    assert store.get_run(first) is None
    assert not store.delete_run(first)


def test_parameterized_inputs_remain_data(store):
    attack = "'); DROP TABLE knowledge_entries; --"
    note = store.add_entry(attack, "Python service", profile_id=attack)
    assert store.list_entries(attack)[0]["id"] == note["id"]
    assert store.list_entries() == []
    assert not store.delete_entry(attack)
    assert store.get_run(attack) is None
    assert store.get_cached_response(attack) is None


@pytest.mark.parametrize("method,args", [
    ("add_entry", ("", "something")),
    ("add_entry", ("title", "")),
    ("add_entry", ("title", "x" * 20_001)),
    ("add_entry", ("x" * 201, "something")),
    ("get_cached_response", ("",)),
])
def test_invalid_text_is_rejected(store, method, args):
    with pytest.raises(ValueError):
        getattr(store, method)(*args)


def test_rejects_invalid_budgets_payloads_and_limits(store):
    with pytest.raises(ValueError):
        store.retrieve("Python", max_chars=-1)
    with pytest.raises(ValueError):
        store.retrieve("Python", max_chars=20_001)
    with pytest.raises(ValueError):
        store.retrieve("Python", limit=101)
    with pytest.raises(ValueError):
        store.list_runs(limit=501)
    with pytest.raises(ValueError):
        store.save_run({"invalid": object()}, job_description="Engineer")
    with pytest.raises(ValueError):
        store.save_run({"number": float("nan")}, job_description="Engineer")
    with pytest.raises(ValueError):
        store.set_cached_response("key", ["not an object"])


def test_cache_round_trip_is_detached_and_persistent(tmp_path):
    path = tmp_path / "cache.sqlite3"
    first = KnowledgeStore(path)
    first.set_cached_response("v2:model:hash", {"nested": [1, 2]})
    copy = first.get_cached_response("v2:model:hash")
    copy["nested"].append(3)
    assert first.get_cached_response("v2:model:hash") == {"nested": [1, 2]}
    first.close()
    reopened = KnowledgeStore(path)
    try:
        assert reopened.get_cached_response("v2:model:hash") == {"nested": [1, 2]}
        assert reopened.get_cached_response("v3:model:hash") is None
    finally:
        reopened.close()


def test_cache_entry_and_byte_budgets_are_enforced(store):
    store.MAX_CACHE_ENTRIES = 2
    store.MAX_CACHE_BYTES = 100
    store.set_cached_response("first", {"payload": "a" * 10})
    store.set_cached_response("second", {"payload": "b" * 10})
    store.set_cached_response("third", {"payload": "c" * 10})
    assert store.get_cached_response("first") is None
    assert store.get_cached_response("second") is not None
    store.set_cached_response("large", {"payload": "z" * 200})
    assert store.get_cached_response("large") is None
    store.set_cached_response("fourth", {"payload": "d" * 60})
    assert store.get_cached_response("second") is None
    assert store.get_cached_response("fourth") is not None


def test_removing_knowledge_invalidates_persisted_responses(store):
    entry = store.add_entry("Python", "Built Python services.")
    store.set_cached_response("old-evidence", {"summary": "Python"})
    assert store.delete_entry(entry["id"])
    assert store.get_cached_response("old-evidence") is None
    assert store.retrieve("Python") == []


def test_one_connection_supports_concurrent_local_requests(store):
    def create(index):
        return store.add_entry(f"Project {index}", f"Python data pipeline number {index}.")

    with ThreadPoolExecutor(max_workers=6) as executor:
        notes = list(executor.map(create, range(30)))
    assert len({note["id"] for note in notes}) == 30
    assert len(store.list_entries()) == 30
    assert len(store.retrieve("Python data", limit=3)) == 3
