"""All tests are offline: real credentials and network connections are forbidden."""

import socket

import pytest
import requests

from app.config import Config


@pytest.fixture(autouse=True)
def offline_tests(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(Config, "ANTHROPIC_API_KEY", "")
    # Every app created by tests has independent local storage.
    monkeypatch.setattr(Config, "KNOWLEDGE_DB_PATH", ":memory:")

    def blocked(*args, **kwargs):
        raise AssertionError("Real network calls are forbidden in tests; provide a mocked response.")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
