"""Application configuration from environment variables."""

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
    DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
    PORT = int(os.getenv("PORT", "5000"))

    ANTHROPIC_API_BASE_URL = os.getenv("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5").strip()
    CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "8192"))
    CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "medium").strip()
    CLAUDE_PLAN_MAX_TOKENS = int(os.getenv("CLAUDE_PLAN_MAX_TOKENS", "6000"))
    CLAUDE_REWRITE_MAX_TOKENS = int(os.getenv("CLAUDE_REWRITE_MAX_TOKENS", "4000"))
    CLAUDE_REPAIR_MAX_TOKENS = int(os.getenv("CLAUDE_REPAIR_MAX_TOKENS", "4000"))
    CLAUDE_PROMPT_CACHING = os.getenv("CLAUDE_PROMPT_CACHING", "1") == "1"
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

    LATEX_COMPILER = os.getenv("LATEX_COMPILER", "").strip()
    LATEX_TIMEOUT_SECONDS = int(os.getenv("LATEX_TIMEOUT_SECONDS", "30"))
    LATEX_MAX_LAYOUT_PASSES = 14
    MAX_CONTENT_LENGTH = 600_000
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"

    MAX_RESUME_CHARS = int(os.getenv("MAX_RESUME_CHARS", "80000"))
    MAX_JD_CHARS = int(os.getenv("MAX_JD_CHARS", "40000"))
    MAX_SUPPLEMENTAL_CHARS = int(os.getenv("MAX_SUPPLEMENTAL_CHARS", "20000"))

    TARGET_SCORE = 95
    MAX_OPTIMIZATION_PASSES = 2
    KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH", str(BASE_DIR / "data" / "knowledge.sqlite3"))
    KNOWLEDGE_MAX_CONTEXT_CHARS = 6000
    KNOWLEDGE_MAX_ENTRIES_PER_RUN = 12

    @classmethod
    def llm_configured(cls) -> bool:
        return bool(cls.ANTHROPIC_API_KEY)


def config_value(name: str):
    """Use the active application's settings, including test/deployment overrides."""
    from flask import current_app, has_app_context

    if has_app_context() and name in current_app.config:
        return current_app.config[name]
    return getattr(Config, name)
