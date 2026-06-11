"""Inference providers: the swappable robot 'brain'."""

from __future__ import annotations

import os
from typing import Any, Callable

from actum.inference.base import InferenceProvider


def resolve_api_key(provider_cfg: dict[str, Any]) -> str:
    """API key for a provider config: explicit key first, then its env var."""
    api_key = str(provider_cfg.get("api_key", "")).strip()
    if api_key:
        return api_key
    env_name = str(provider_cfg.get("api_key_env", "")).strip()
    return os.environ.get(env_name, "") if env_name else ""


def build_provider(
    settings: Any,
    *,
    compute: str,
    pop_pending_frame: Callable[[], str | None],
    resolve_model_path: Callable[[], str],
) -> InferenceProvider:
    """Construct the active inference provider from runtime settings.

    The selected provider comes from ``models.active_provider``; its model id and
    (for cloud providers) API key come from that provider's config entry.
    """
    models = settings.to_config(include_secrets=True)["models"]
    provider = str(models.get("active_provider", "local")).lower().strip()
    provider_cfg = models.get("providers", {}).get(provider, {})
    model = str(provider_cfg.get("model", "")).strip()

    if provider in {"local", "litert"}:
        from actum.inference.litert import LiteRTProvider

        return LiteRTProvider(
            model=model,
            compute=compute,
            pop_pending_frame=pop_pending_frame,
            resolve_model_path=resolve_model_path,
        )

    if provider == "openai":
        from actum.inference.openai import OpenAIProvider

        return OpenAIProvider(
            model=model,
            pop_pending_frame=pop_pending_frame,
            api_key=resolve_api_key(provider_cfg),
        )

    if provider == "anthropic":
        raise RuntimeError(
            "The anthropic provider is not implemented yet. Use 'local' or 'openai' "
            "in models.active_provider."
        )

    raise RuntimeError(f"Unknown inference provider: {provider!r}")


__all__ = ["InferenceProvider", "build_provider", "resolve_api_key"]
