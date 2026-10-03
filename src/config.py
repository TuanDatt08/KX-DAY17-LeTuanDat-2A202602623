from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Student TODO: load environment variables and return a LabConfig.

    Pseudocode:
    1. Resolve the repo root or default to the current file parent.
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass  # python-dotenv is optional; plain env vars still work

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_from_env("LLM")
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        # ~600 tokens: stress turns (~200 tokens each) compact several times,
        # standard turns (~15 tokens) rarely do -> shows the short vs long trade-off.
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=_provider_from_env("JUDGE", fallback=model),
    )


API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
BASE_URL_ENV = {"custom": "CUSTOM_BASE_URL", "ollama": "OLLAMA_BASE_URL"}


def _provider_from_env(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Read `<PREFIX>_PROVIDER/_MODEL/_TEMPERATURE`; judge falls back to the main model."""

    provider = normalize_provider(
        os.getenv(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai")
    )
    key_env = API_KEY_ENV.get(provider)
    url_env = BASE_URL_ENV.get(provider)
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL") or (fallback.model_name if fallback else "gpt-4o-mini"),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=os.getenv(key_env) if key_env else None,
        base_url=os.getenv(url_env) if url_env else None,
    )
