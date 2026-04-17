from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROVIDER_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class MissingAPIKeyError(RuntimeError):
    """Raised when the LLM provider's API key env var is not set."""


def load_dotenv(path: str | Path = ".env") -> bool:
    """Load ``KEY=VALUE`` pairs from a .env file into ``os.environ``.

    Existing environment variables are preserved so CI-exported values
    take precedence. Returns True if the file was loaded.
    """
    env_path = Path(path)
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def infer_provider(model: str) -> str | None:
    """Guess the LLM provider from a model identifier."""
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return None


def check_api_key(adapter_config: dict[str, Any]) -> None:
    """Ensure the API key for the configured LLM is available.

    Looks up the provider from ``adapter_config['llm_provider']``, or
    infers it from ``adapter_config['llm_model']`` when the provider
    is not set. No-ops when neither is present (e.g. adapters that
    don't call an LLM directly).
    """
    provider = adapter_config.get("llm_provider")
    model = adapter_config.get("llm_model")

    if not provider and model:
        provider = infer_provider(str(model))

    if not provider:
        return

    env_var = PROVIDER_ENV_VARS.get(str(provider).lower())
    if env_var is None:
        return

    if not os.environ.get(env_var):
        model_hint = f" (llm_model={model!r})" if model else ""
        raise MissingAPIKeyError(
            f"Missing API key for provider {provider!r}{model_hint}: "
            f"environment variable {env_var} is not set. "
            f"Export it in your shell or add it to a .env file in the current "
            f"directory. See .env.example for the expected variables."
        )
