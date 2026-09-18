"""Environment-driven configuration.

No secret values live in this file. Every credential is read from the
environment (see .env.example for the variable names).
"""

import os


def _as_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _as_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    # ---- service ----
    API_PORT = _as_int("API_PORT", 5000)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # ---- LLM provider ----
    # "openai"    -> OpenAI or any OpenAI-compatible endpoint (Groq, OpenRouter,
    #                Together, vLLM, Ollama) by also setting LLM_BASE_URL.
    # "anthropic" -> Anthropic Messages API.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

    # Deterministic extraction matters more than creative wording here.
    LLM_TEMPERATURE = _as_float("LLM_TEMPERATURE", 0.0)
    LLM_MAX_TOKENS = _as_int("LLM_MAX_TOKENS", 900)

    # Judge timeout is 30s per request and p95 <= 5s earns full latency points,
    # so the model call is kept on a short leash with one fast retry.
    LLM_TIMEOUT_SECONDS = _as_float("LLM_TIMEOUT_SECONDS", 12.0)
    LLM_MAX_RETRIES = _as_int("LLM_MAX_RETRIES", 1)

    # Safety net only: if the LLM errors out or returns unusable JSON, a
    # deterministic parser keeps the service responding with a valid schedule
    # instead of crashing. It is never the primary interpretation path.
    ENABLE_RULE_FALLBACK = os.getenv("ENABLE_RULE_FALLBACK", "true").lower() != "false"

    # Run the judge-style replay on every response and log any violation.
    SELF_CHECK = os.getenv("SELF_CHECK", "true").lower() != "false"

    @property
    def llm_configured(self) -> bool:
        return bool(self.LLM_API_KEY) or self.LLM_BASE_URL is not None


config = Config()
