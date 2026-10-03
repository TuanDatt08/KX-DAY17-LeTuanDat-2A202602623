from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


SUPPORTED_PROVIDERS = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
PROVIDER_ALIASES = {
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "google": "gemini",
    "google_genai": "gemini",
    "openai_compatible": "custom",
}


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic` and reject unknown providers."""

    provider = (value or "openai").strip().lower()
    provider = PROVIDER_ALIASES.get(provider, provider)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {value!r}. Choose one of {sorted(SUPPORTED_PROVIDERS)}")
    return provider


def has_credentials(config: ProviderConfig) -> bool:
    """Live mode is possible only with an API key (or a local Ollama server)."""

    return normalize_provider(config.provider) == "ollama" or bool(config.api_key)


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`

    Imports are lazy so offline mode works without every provider SDK installed.
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=config.api_key, base_url=config.base_url, **common)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(google_api_key=config.api_key, **common)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(api_key=config.api_key, **common)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(base_url=config.base_url or "http://localhost:11434", **common)

    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(api_key=config.api_key, **common)
