"""Anthropic Messages API adapter. API credentials stay on the backend."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ValidationError

from app.config import config_value
from app.llm.outputs import api_schema

SAFE_LOG = logging.getLogger("resumeopt.llm")


class LLMError(Exception):
    """Safe actionable error without provider bodies or submitted documents."""


class LLMClient:
    def __init__(self, api_base: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: int | None = None,
                 max_retries: int | None = None, max_tokens: int | None = None):
        self.api_base = (api_base or config_value("ANTHROPIC_API_BASE_URL")).rstrip("/")
        self.api_key = api_key if api_key is not None else config_value("ANTHROPIC_API_KEY")
        self.model = model or config_value("CLAUDE_MODEL")
        self.timeout = timeout if timeout is not None else config_value("LLM_TIMEOUT_SECONDS")
        self.max_retries = max_retries if max_retries is not None else config_value("LLM_MAX_RETRIES")
        self.max_tokens = max_tokens if max_tokens is not None else config_value("CLAUDE_MAX_TOKENS")
        self.usage = {"input_tokens": 0, "output_tokens": 0,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                      "api_calls": 0}

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def chat_json(self, system: str, user: str, *, temperature: float = 0.2,
                  output_model: type[BaseModel] | None = None,
                  max_tokens: int | None = None) -> dict[str, Any]:
        """Request native structured JSON and reject incomplete or invalid results."""
        if not self.configured:
            raise LLMError("Set ANTHROPIC_API_KEY in .env to enable Claude optimization.")
        base = urlsplit(self.api_base)
        if base.scheme != "https" or not base.netloc or base.username or base.password:
            raise LLMError("ANTHROPIC_API_BASE_URL must be an HTTPS URL without credentials.")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": min(self.max_tokens, max_tokens) if max_tokens is not None else self.max_tokens,
            "system": ([{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}]
                       if config_value("CLAUDE_PROMPT_CACHING") else system),
            "messages": [{"role": "user", "content": user}],
        }
        if payload["max_tokens"] < 1:
            raise LLMError("Claude's output budget must be a positive number.")
        effort = config_value("CLAUDE_EFFORT")
        if effort:
            if effort not in {"low", "medium", "high"}:
                raise LLMError("CLAUDE_EFFORT must be low, medium, high, or empty.")
            payload["output_config"] = {"effort": effort}
        # Newer Claude models manage sampling automatically.
        if output_model is not None:
            payload.setdefault("output_config", {})["format"] = {
                "type": "json_schema", "schema": api_schema(output_model),
            }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        retries = max(0, min(int(self.max_retries), 3))
        for attempt in range(retries + 1):
            try:
                self.usage["api_calls"] += 1
                response = requests.post(
                    f"{self.api_base}/messages", headers=headers, json=payload,
                    timeout=(10, self.timeout), allow_redirects=False,
                )
            except requests.RequestException:
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise LLMError("Could not reach Claude. Check your connection and try again.") from None

            status = response.status_code
            if status in {408, 409, 429, 500, 502, 503, 504, 529} and attempt < retries:
                try:
                    delay = float(response.headers.get("retry-after", 2 ** attempt))
                except (ValueError, TypeError):
                    delay = 2 ** attempt
                SAFE_LOG.warning("Claude HTTP %s; retry %s", status, attempt + 1)
                time.sleep(max(0, min(delay, 15)))
                continue
            if not 200 <= status < 300:
                messages = {
                    400: "Claude rejected the request. Check CLAUDE_MODEL and structured-output support.",
                    401: "Claude authentication failed. Check ANTHROPIC_API_KEY in .env.",
                    403: "This Claude API key does not have access to the selected model.",
                    404: "Claude model or endpoint not found. Check CLAUDE_MODEL and ANTHROPIC_API_BASE_URL.",
                    429: "Claude rate limit reached. Wait a moment and try again.",
                }
                raise LLMError(messages.get(status, f"Claude is unavailable (HTTP {status}). Try again later."))

            try:
                data = response.json()
                usage = data.get("usage", {})
                if isinstance(usage, dict):
                    for key in self.usage:
                        value = usage.get(key)
                        if key != "api_calls" and type(value) is int and value >= 0:
                            self.usage[key] += value
                stop = data.get("stop_reason")
                if stop == "max_tokens":
                    raise LLMError("Claude's response was cut off. Increase CLAUDE_MAX_TOKENS and try again.")
                if stop == "refusal":
                    raise LLMError("Claude declined this request. Review the submitted content and try again.")
                if stop != "end_turn":
                    raise LLMError("Claude did not finish a complete response. Please try again.")
                content = "".join(
                    block["text"] for block in data["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                parsed = parse_json_object(content)
                if output_model is not None:
                    parsed = output_model.model_validate_json(json.dumps(parsed)).model_dump(mode="json")
                return parsed
            except ValidationError as exc:
                # Field paths/types only. Pydantic's full error string includes
                # rejected values, which may contain private submitted content.
                SAFE_LOG.warning("Claude %s validation failed at %s", output_model.__name__, [
                    {"path": error["loc"], "type": error["type"]}
                    for error in exc.errors(include_input=False, include_context=False)
                ])
                raise LLMError("Claude returned an invalid structured response. Please try again.") from None
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                SAFE_LOG.warning("Claude %s response failed (%s)",
                                 output_model.__name__ if output_model else "JSON", type(exc).__name__)
                raise LLMError("Claude returned an invalid structured response. Please try again.") from None
        raise LLMError("Claude is unavailable. Try again later.")


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse JSON objects, tolerating markdown wrappers in older fixtures."""
    if not isinstance(content, str) or not content.strip():
        raise json.JSONDecodeError("Empty content", "", 0)
    text = content.strip()
    if text.startswith(chr(96) * 3):
        text = re.sub(r"^" + chr(96) * 3 + r"(?:json)?\s*", "", text)
        text = re.sub(r"\s*" + chr(96) * 3 + r"$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise json.JSONDecodeError("Expected JSON object", text, 0)
    return data


def get_llm_client() -> LLMClient:
    return LLMClient()

