"""Native JSON schemas must preserve data fields while removing metadata."""

from app.llm.outputs import EvidenceOutput, JobOutput, PlanOutput, RewriteOutput, api_schema


def test_schema_preserves_the_actual_title_property():
    role_schema = api_schema(RewriteOutput)["$defs"]["RewrittenRole"]
    assert role_schema["properties"]["title"] == {"type": "string"}
    assert "title" in role_schema["required"]


def test_all_required_fields_are_declared_in_native_schema():
    def check(node):
        if isinstance(node, list):
            for child in node:
                check(child)
        elif isinstance(node, dict):
            if node.get("type") == "object":
                assert set(node.get("required", [])) <= set(node.get("properties", {}))
                assert node.get("additionalProperties") is False
            for child in node.values():
                check(child)

    for model in (JobOutput, EvidenceOutput, PlanOutput, RewriteOutput):
        check(api_schema(model))
