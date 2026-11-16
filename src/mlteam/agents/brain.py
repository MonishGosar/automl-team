"""Connect to a model through Tau's own provider settings.

mlteam does not manage API keys. Whatever provider you logged into with
``tau`` (``/login``) is reused here, so there is one place to configure models.
"""

from __future__ import annotations

from dataclasses import dataclass

from tau_agent.provider import ModelProvider


@dataclass(frozen=True)
class Brain:
    provider: ModelProvider
    model: str
    label: str


def resolve_brain(provider_name: str | None = None, model: str | None = None) -> tuple[Brain | None, str]:
    """Return (brain, note). ``brain`` is None when no usable model is
    configured; ``note`` then says what to do about it."""
    try:
        from tau_coding.credentials import FileCredentialStore
        from tau_coding.provider_config import (
            load_provider_settings,
            provider_has_usable_credentials,
            resolve_provider_selection,
        )
        from tau_coding.provider_runtime import create_model_provider

        settings = load_provider_settings()
        selection = resolve_provider_selection(settings, provider_name=provider_name, model=model)
        credentials = FileCredentialStore()
        if not provider_has_usable_credentials(selection.provider, credential_reader=credentials):
            return None, (
                f"No credentials for provider '{selection.provider.name}'. "
                "Run `tau`, then /login, to connect a model."
            )
        provider = create_model_provider(
            selection.provider, credential_store=credentials, model=selection.model
        )
    except Exception as exc:  # noqa: BLE001 - any config problem means "no model", not a crash
        return None, f"No model connected ({exc}). Run `tau`, then /login."
    return Brain(provider=provider, model=selection.model, label=f"{selection.provider.name}/{selection.model}"), "ok"
