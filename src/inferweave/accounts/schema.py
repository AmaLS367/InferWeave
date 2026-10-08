"""Credential shapes required by each provider that supports account pools."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from credweave import Credential

from inferweave.core.exceptions import AccountConfigurationError


@dataclass(frozen=True)
class CredentialSchema:
    """Secret fields and nonsecret metadata accepted for one provider's accounts."""

    provider: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    paired: tuple[tuple[str, str], ...] = ()
    """Optional secrets that must be configured together (both or neither)."""
    metadata: tuple[str, ...] = ()
    """Provider-specific nonsecret metadata keys (scheduling keys are always allowed)."""

    def validate_secret_names(self, account_id: str, names: set[str]) -> None:
        missing = [name for name in self.required if name not in names]
        unknown = sorted(names - set(self.required) - set(self.optional))
        if missing:
            raise AccountConfigurationError(
                f"{self.provider} account '{account_id}' is missing required secret(s): "
                f"{', '.join(missing)}."
            )
        if unknown:
            raise AccountConfigurationError(
                f"{self.provider} account '{account_id}' has unsupported secret(s): "
                f"{', '.join(unknown)}. Supported: "
                f"{', '.join(self.required + self.optional)}."
            )

    def validate_metadata(self, account_id: str, metadata: Mapping[str, Any]) -> None:
        unknown = sorted(set(metadata) - set(self.metadata) - SCHEDULING_METADATA)
        if unknown:
            raise AccountConfigurationError(
                f"{self.provider} account '{account_id}' has unsupported metadata: "
                f"{', '.join(unknown)}. Supported: "
                f"{', '.join(sorted(set(self.metadata) | SCHEDULING_METADATA))}."
            )
        for key, value in metadata.items():
            if key in self.metadata and value is not None and not isinstance(value, str):
                raise AccountConfigurationError(
                    f"{self.provider} account '{account_id}': metadata '{key}' must be a string."
                )

    def validate_credential(self, credential: Credential) -> None:
        """Validates a credential loaded at runtime (e.g. from a hot-reloaded JSON file)."""
        names = set(credential.secret_keys)
        self.validate_secret_names(credential.id, names)
        for name in names:
            value = credential.get_secret(name)
            if not isinstance(value, str) or not value.strip():
                raise AccountConfigurationError(
                    f"{self.provider} account '{credential.id}': secret '{name}' must be a nonempty string."
                )
        for first, second in self.paired:
            if (first in names) != (second in names):
                raise AccountConfigurationError(
                    f"{self.provider} account '{credential.id}' must configure '{first}' and "
                    f"'{second}' together."
                )
        self.validate_metadata(credential.id, credential.metadata)


SCHEDULING_METADATA = frozenset({"weight", "priority", "max_concurrency", "tags"})
"""CredWeave scheduling metadata understood by the built-in strategies."""

MODAL = CredentialSchema(
    provider="modal",
    required=("token_id", "token_secret"),
    optional=("proxy_token_id", "proxy_token_secret"),
    paired=(("proxy_token_id", "proxy_token_secret"),),
    metadata=("environment",),
)
LIGHTNING = CredentialSchema(
    provider="lightning",
    required=("user_id", "api_key"),
    metadata=("teamspace",),
)
RUNPOD = CredentialSchema(provider="runpod", required=("api_key",))
VAST = CredentialSchema(provider="vast", required=("api_key",))

SCHEMAS: dict[str, CredentialSchema] = {s.provider: s for s in (MODAL, LIGHTNING, RUNPOD, VAST)}


def schema_for(provider: str) -> CredentialSchema:
    try:
        return SCHEMAS[provider.lower()]
    except KeyError:
        raise AccountConfigurationError(
            f"Account pools are not supported for provider '{provider}'. Supported providers: "
            f"{', '.join(SCHEMAS)}. Other providers use their native default credentials."
        ) from None
