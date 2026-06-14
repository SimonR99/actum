"""Runtime settings for model/provider/tool selection."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

DEFAULT_PROVIDER_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

STT_ENGINES = ("whisper", "openai", "model")

DEFAULT_SPEECH = {
    "stt_engine": "whisper",
    "whisper_model": "base",
    "whisper_compute": "auto",
}


class RuntimeSettings:
    def __init__(self, config: dict[str, Any], capability_names: list[str]):
        self._capability_names = list(capability_names)
        self.models = _default_models()
        raw_models = (
            config.get("models", {}) if isinstance(config.get("models"), dict) else {}
        )
        self.models = _merge_dict(self.models, raw_models)

        raw_tools = (
            config.get("tools", {}) if isinstance(config.get("tools"), dict) else {}
        )
        configured = raw_tools.get("enabled")
        if isinstance(configured, list) and configured:
            self.enabled_tools = {str(item) for item in configured}
        else:
            self.enabled_tools = set(capability_names)
        disabled = raw_tools.get("disabled")
        if isinstance(disabled, list):
            self.enabled_tools.difference_update(str(item) for item in disabled)

        for provider, env_name in DEFAULT_PROVIDER_ENVS.items():
            provider_cfg = self.models.setdefault("providers", {}).setdefault(
                provider, {}
            )
            provider_cfg.setdefault("api_key_env", env_name)
            if os.environ.get(str(provider_cfg["api_key_env"])):
                provider_cfg["api_key_configured"] = True

        raw_speech = (
            config.get("speech", {}) if isinstance(config.get("speech"), dict) else {}
        )
        self.speech = {**DEFAULT_SPEECH, **raw_speech}
        if str(self.speech.get("stt_engine")) not in STT_ENGINES:
            self.speech["stt_engine"] = DEFAULT_SPEECH["stt_engine"]
        self.language = str(config.get("language", "en")).strip().lower()

    def set_model_provider(
        self,
        provider: str,
        model: str = "",
        api_key: str = "",
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        name = str(provider).strip().lower()
        if not name:
            raise ValueError("provider is required")
        providers = self.models.setdefault("providers", {})
        cfg = providers.setdefault(name, {"model": "", "enabled": False})
        if model:
            cfg["model"] = str(model).strip()
        if enabled is not None:
            cfg["enabled"] = bool(enabled)
        if api_key:
            env_name = str(
                cfg.get("api_key_env")
                or DEFAULT_PROVIDER_ENVS.get(name)
                or f"{name.upper()}_API_KEY"
            )
            cfg["api_key_env"] = env_name
            cfg["api_key_configured"] = True
            cfg["_api_key"] = api_key
            os.environ[env_name] = api_key
        self.models["active_provider"] = name
        return self.to_dict()["models"]

    def set_stt_engine(self, engine: str) -> dict[str, Any]:
        name = str(engine).strip().lower()
        if name not in STT_ENGINES:
            raise ValueError(
                f"Unknown STT engine {engine!r}. Choose one of: {', '.join(STT_ENGINES)}"
            )
        self.speech["stt_engine"] = name
        return dict(self.speech)

    def set_tool_enabled(self, tool: str, enabled: bool):
        name = str(tool).strip()
        if not name:
            raise ValueError("tool is required")
        if enabled:
            self.enabled_tools.add(name)
        else:
            self.enabled_tools.discard(name)

    def tool_enabled(self, name: str) -> bool:
        return str(name) in self.enabled_tools

    def to_dict(self) -> dict[str, Any]:
        models = deepcopy(self.models)
        for cfg in models.get("providers", {}).values():
            cfg.pop("_api_key", None)
            cfg.pop("api_key", None)
            env_name = cfg.get("api_key_env")
            cfg["api_key_configured"] = bool(
                cfg.get("api_key_configured")
                or (env_name and os.environ.get(str(env_name)))
            )
        return {
            "language": self.language,
            "models": models,
            "tools": {
                "enabled": sorted(self.enabled_tools),
                "available": sorted(self._capability_names),
            },
            "speech": {**self.speech, "available_stt": list(STT_ENGINES)},
        }

    def to_config(self, include_secrets: bool = False) -> dict[str, Any]:
        settings = self.to_dict()
        if include_secrets:
            for provider, cfg in self.models.get("providers", {}).items():
                secret = cfg.get("_api_key")
                if secret:
                    settings["models"]["providers"].setdefault(provider, {})[
                        "api_key"
                    ] = secret
        return settings


def _default_models() -> dict[str, Any]:
    return {
        "active_provider": "local",
        "providers": {
            "local": {
                "enabled": True,
                "model": "litert-community/gemma-4-E2B-it-litert-lm",
                "api_key_configured": False,
            },
            "openai": {
                "enabled": False,
                "model": "",
                "api_key_env": "OPENAI_API_KEY",
                "api_key_configured": False,
            },
            "anthropic": {
                "enabled": False,
                "model": "",
                "api_key_env": "ANTHROPIC_API_KEY",
                "api_key_configured": False,
            },
        },
    }


def _merge_dict(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged
