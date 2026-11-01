"""Malformed model response handling."""

import json

import pytest

from app.llm.adapter import parse_json_object


def test_parse_plain_json():
    data = parse_json_object('{"a": 1, "b": [2]}')
    assert data["a"] == 1
    assert data["b"] == [2]


def test_parse_fenced_json():
    content = """```json
{"role_title": "Engineer", "requirements": []}
```"""
    data = parse_json_object(content)
    assert data["role_title"] == "Engineer"


def test_parse_embedded_object():
    content = "Here you go:\n{\"ok\": true, \"n\": 3}\nThanks"
    data = parse_json_object(content)
    assert data["ok"] is True


def test_reject_non_object():
    with pytest.raises(json.JSONDecodeError):
        parse_json_object("[1, 2, 3]")


def test_reject_empty():
    with pytest.raises(json.JSONDecodeError):
        parse_json_object("   ")


def test_reject_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_json_object("not json at all")
