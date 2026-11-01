"""Demo mode helpers — clearly labeled, never presented as live LLM analysis."""

DEMO_NOTICE = (
    "DEMO MODE: No Claude API key is configured. "
    "This run uses deterministic local analysis on your inputs (and/or sample data). "
    "It is NOT a live model analysis and must not be treated as one. "
    "Set ANTHROPIC_API_KEY (and optionally CLAUDE_MODEL) in .env for Claude optimization."
)


def is_demo_mode() -> bool:
    from app.llm.adapter import get_llm_client

    return not get_llm_client().configured
