"""Per-run usage and versioned, exact-input response reuse; no embedding calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3

from pydantic import BaseModel, ValidationError

from app.config import config_value
from app.llm.adapter import LLMClient

PROMPT_VERSION = "knowledge-plan-v1"


class OptimizationSession:
    def __init__(self, client: LLMClient, store=None, profile_id: str = "default"):
        self.client = client
        self.store = store
        self.profile_id = profile_id
        self.before = dict(getattr(client, "usage", {}))
        self.cache_hits = 0

    def request(self, stage: str, system: str, user: str,
                output_model: type[BaseModel], max_tokens: int) -> dict:
        signature = {
            "version": PROMPT_VERSION, "stage": stage, "profile": self.profile_id,
            "endpoint": getattr(self.client, "api_base", ""),
            "model": getattr(self.client, "model", ""),
            "effort": config_value("CLAUDE_EFFORT"),
            "schema": output_model.model_json_schema(), "system": system,
            "user": user, "max_tokens": min(getattr(self.client, "max_tokens", max_tokens), max_tokens),
        }
        key = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode()).hexdigest()
        if self.store is not None:
            try:
                cached = self.store.get_cached_response(key)
                if cached is not None:
                    value = output_model.model_validate_json(json.dumps(cached)).model_dump(mode="json")
                    self.cache_hits += 1
                    return value
            except (ValidationError, ValueError, TypeError, sqlite3.Error, OSError):
                pass  # A stale/corrupt cache must not prevent a fresh request.
        value = self.client.chat_json(system, user, output_model=output_model, max_tokens=max_tokens)
        value = output_model.model_validate_json(json.dumps(value)).model_dump(mode="json")
        if self.store is not None:
            try:
                self.store.set_cached_response(key, value)
            except (sqlite3.Error, OSError, ValueError):
                pass
        return value

    def usage(self) -> dict:
        current = getattr(self.client, "usage", {})
        values = {key: max(0, current.get(key, 0) - self.before.get(key, 0))
                  for key in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                              "cache_creation_input_tokens", "api_calls")}
        values.update(local_cache_hits=self.cache_hits, model=getattr(self.client, "model", ""))
        values["total_input_tokens"] = sum(values[key] for key in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        return values
