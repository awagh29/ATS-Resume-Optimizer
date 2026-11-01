"""Knowledge origins, multiline attribution, and derived prompt budgets, offline."""

import json

import pytest

from app.config import Config
from app.knowledge import KnowledgeStore
from app.llm.adapter import LLMClient
from app.llm.outputs import JobOutput
from app.models.schemas import CompilationResult
from app.pipeline.orchestrator import run_optimization
from app.pipeline.parse_resume import parse_resume_latex


RESUME = r"""\begin{document}
Alex Example
\section{Experience}
\resumeSubheading{BrightPath Labs}{2021 -- Present}{Software Engineer}{Seattle, WA}
\resumeItem{Built Python REST APIs for analytics dashboards.}
\resumeSubheading{Other Co}{2019 -- 2021}{Software Engineer}{Seattle, WA}
\resumeItem{Maintained SQL reports.}
\section{Projects}
\textbf{Data Explorer}
\resumeItem{Built a Python analytics dashboard.}
\section{Skills}
\textbf{Languages:} Python, SQL
\end{document}"""


@pytest.fixture
def store():
    value = KnowledgeStore(":memory:")
    yield value
    value.close()


@pytest.fixture
def planning_payloads(monkeypatch):
    captured = []
    monkeypatch.setattr(Config, "MAX_OPTIMIZATION_PASSES", 1)

    def forbidden(*args, **kwargs):
        pytest.fail("Provenance tests must never call a model provider")

    def fake_request(self, stage, system, user, output_model, max_tokens):
        payload = json.loads(user)
        if stage == "plan":
            captured.append(payload)
            analysis = {key: [] for key in JobOutput.model_fields}
            analysis.update(role_title="Backend Engineer", seniority="Mid-level")
            return {"analysis": analysis, "mappings": []}
        source = payload["resume_facts"]
        return {
            "summary": "", "experience": [{
                "source_fact_id": role["fact_id"], "employer": role["employer"],
                "title": role["title"], "functional_descriptor": "", "bullets": [
                    {"text": bullet["text"], "fact_ids": [bullet["fact_id"]]}
                    for bullet in role["bullets"]
                ],
            } for role in source["experience"]],
            "projects": [], "skills": [], "content_decisions": [], "clarification_questions": [],
        }

    def fit(resume, *, page_size):
        return resume, "offline LaTeX", CompilationResult(error="No compiler needed for this check"), []

    monkeypatch.setattr(LLMClient, "chat_json", forbidden)
    monkeypatch.setattr("app.pipeline.orchestrator.OptimizationSession.request", fake_request)
    monkeypatch.setattr("app.pipeline.orchestrator.fit_resume_to_one_page", fit)
    return captured


def optimize(store, **kwargs):
    return run_optimization(RESUME, "Required: Python SQL API latency dashboards.",
                            llm=LLMClient(api_key="offline-placeholder"), knowledge_store=store,
                            **kwargs)


def test_skill_matching_a_project_name_is_still_a_skill(store, planning_payloads):
    entry = store.add_entry("Python skill", "Python", category="skill")
    result = run_optimization(RESUME.replace("Data Explorer", "Python"),
                              "Required: Python", llm=LLMClient(api_key="offline-placeholder"),
                              knowledge_store=store)
    assert result.knowledge_used[0]["id"] == entry["id"]
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    assert records[0]["text"] == "[CONFIRMED Skill] Python"


def test_duplicate_knowledge_and_form_text_keep_independent_origins(store, planning_payloads):
    first = store.add_entry("Python skill one", "Python and SQL", "skill")
    second = store.add_entry("Python skill two", "Python and SQL", "skill")
    result = optimize(store, supplemental_text="Skill: Python and SQL")
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    assert len(records) == 3
    assert "knowledge_id" not in records[0]
    assert records[0]["fact_id"] == "supplemental-1"
    assert {record["knowledge_id"] for record in records[1:]} == {first["id"], second["id"]}
    used = {entry["id"]: entry for entry in result.knowledge_used}
    assert len(used[first["id"]]["fact_ids"]) == 1
    assert len(used[second["id"]]["fact_ids"]) == 1
    assert used[first["id"]]["fact_ids"] != used[second["id"]]["fact_ids"]
    assert all("supplemental-1" not in entry["fact_ids"] for entry in result.knowledge_used)
    assert store.get_run(result.run_id)["result"]["knowledge_used"] == result.knowledge_used


def test_known_employer_header_scopes_complete_accomplishment_lines(store, planning_payloads):
    content = "At BrightPath Labs:\nReduced Python API latency by 47%.\nBuilt Python integration tests."
    note = store.add_entry("Python performance", content)
    result = optimize(store)
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    assert [record["text"] for record in records] == [
        "[CONFIRMED Accomplishment] At BrightPath Labs: Reduced Python API latency by 47%.",
        "[CONFIRMED Accomplishment] At BrightPath Labs: Built Python integration tests.",
    ]
    assert result.knowledge_used == [{
        "id": note["id"], "title": note["title"], "category": note["category"],
        "content": content, "fact_ids": [record["fact_id"] for record in records],
    }]


def test_explicit_other_employer_overrides_carried_scope_without_metric_leakage(store, planning_payloads):
    store.add_entry("Python API work", "At BrightPath Labs:\n"
                    "At Other Co, reduced API latency by 47%.\nBuilt Python tests.")
    optimize(store)
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    assert records[0]["text"] == "[CONFIRMED Accomplishment] At Other Co, reduced API latency by 47%."
    assert "BrightPath Labs" not in records[0]["text"]
    assert records[1]["text"] == "[CONFIRMED Accomplishment] At BrightPath Labs: Built Python tests."


def test_new_known_header_changes_scope_for_following_facts(store, planning_payloads):
    store.add_entry("Python APIs", "At BrightPath Labs:\nBuilt Python APIs.\n"
                    "At Other Co:\nReduced SQL report latency by 47%.")
    optimize(store)
    texts = [record["text"] for record in planning_payloads[0]["confirmed_supplemental_facts"]]
    assert texts == [
        "[CONFIRMED Accomplishment] At BrightPath Labs: Built Python APIs.",
        "[CONFIRMED Accomplishment] At Other Co: Reduced SQL report latency by 47%.",
    ]


def test_explicit_source_id_overrides_other_employer_header(store, planning_payloads):
    source = parse_resume_latex(RESUME).experience[1]
    store.add_entry("Python API notes", "At BrightPath Labs:\n"
                    f"For {source.fact_id}, reduced Python API latency by 47%.")
    optimize(store)
    text = planning_payloads[0]["confirmed_supplemental_facts"][0]["text"]
    assert source.fact_id in text
    assert "BrightPath Labs" not in text


def test_project_header_and_explicit_project_attribution_are_preserved(store, planning_payloads):
    store.add_entry("Python project notes", "Project: Data Explorer:\n"
                    "Built Python dashboards.\nAt BrightPath Labs:\n"
                    "For Data Explorer, reduced Python API latency by 47%.")
    optimize(store)
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    assert records[0]["text"] == "[CONFIRMED Accomplishment] Project: Data Explorer: Built Python dashboards."
    assert records[1]["text"] == "[CONFIRMED Accomplishment] For Data Explorer, reduced Python API latency by 47%."
    assert "BrightPath Labs" not in records[1]["text"]


def test_scopes_do_not_leak_between_independent_knowledge_entries(store, planning_payloads, monkeypatch):
    scoped = store.add_entry("Python scoped", "At BrightPath Labs:\nBuilt Python tests.")
    unscoped = store.add_entry("Python unscoped", "Reduced Python latency by 47%.")
    monkeypatch.setattr(store, "retrieve", lambda *args, **kwargs: [scoped, unscoped])
    optimize(store)
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    assert "BrightPath Labs" in records[0]["text"]
    assert records[1]["text"] == "[CONFIRMED Accomplishment] Reduced Python latency by 47%."


def test_derived_record_budget_includes_ids_labels_and_escaping(store, planning_payloads, monkeypatch):
    monkeypatch.setattr(Config, "KNOWLEDGE_MAX_CONTEXT_CHARS", 650)
    content = "\n".join(['Python SQL "API" tests.'] * 12)
    entry = store.add_entry("Python SQL", content, "skill")
    result = optimize(store, supplemental_text="Metric: Current form facts stay intact 9%.")
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    derived = [record for record in records if "knowledge_id" in record]
    serialized = json.dumps(derived, ensure_ascii=False, separators=(",", ":"))
    assert len(serialized) <= 650
    assert 0 < len(derived) < 12
    assert all(record["text"] == '[CONFIRMED Skill] Python SQL "API" tests.' for record in derived)
    assert records[0]["text"] == "[CONFIRMED Metric] Current form facts stay intact 9%."
    assert "knowledge_id" not in records[0]
    assert any("budget" in warning for warning in result.warnings)
    assert result.knowledge_used[0]["id"] == entry["id"]
    assert result.knowledge_used[0]["content"] == content
    assert result.knowledge_used[0]["fact_ids"] == [record["fact_id"] for record in derived]


def test_budget_omission_excludes_unused_entries_from_provenance(store, planning_payloads, monkeypatch):
    first = store.add_entry("Python accepted", "Python", "skill")
    second = store.add_entry("Python omitted", "SQL", "skill")
    monkeypatch.setattr(store, "retrieve", lambda *args, **kwargs: [first, second])
    monkeypatch.setattr(Config, "KNOWLEDGE_MAX_CONTEXT_CHARS", 140)
    result = optimize(store)
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    assert len(records) == 1
    assert records[0]["knowledge_id"] == first["id"]
    assert [entry["id"] for entry in result.knowledge_used] == [first["id"]]
    assert any("budget" in warning for warning in result.warnings)


def test_final_ids_follow_confirmation_category_order_without_provenance_collision(store, planning_payloads, monkeypatch):
    accomplishment = store.add_entry("Python accomplishment", "At BrightPath Labs, built Python APIs.")
    metric = store.add_entry("Python metric", "At Other Co, reduced Python latency by 47%.", "metric")
    monkeypatch.setattr(store, "retrieve", lambda *args, **kwargs: [accomplishment, metric])
    result = optimize(store, supplemental_text="Skill: Python")
    records = planning_payloads[0]["confirmed_supplemental_facts"]
    assert [(record["fact_id"], record.get("knowledge_id")) for record in records] == [
        ("supplemental-1", metric["id"]),
        ("supplemental-2", accomplishment["id"]),
        ("supplemental-3", None),
    ]
    sources = {entry["id"]: entry["fact_ids"] for entry in result.knowledge_used}
    assert sources == {accomplishment["id"]: ["supplemental-2"], metric["id"]: ["supplemental-1"]}
