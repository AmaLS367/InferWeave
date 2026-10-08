"""Resolved provider accounts handed to compute providers.

A :class:`ProviderAccount` is the only object through which a provider adapter sees secret
material. It wraps a CredWeave :class:`~credweave.Credential` (which already masks secret values
in ``repr``/``str``) and never renders secrets itself.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from credweave import Credential

from inferweave.accounts.schema import schema_for
from inferweave.core.exceptions import ProviderAuthError
from inferweave.domain.deployment_record import AMBIENT_ACCOUNT

AMBIENT_ACCOUNT_ID = AMBIENT_ACCOUNT
"""Account id of a provider's native default credentials (environment, provider config files).

Used when no account pool is configured for a provider. Deployments created this way are bound
to the ambient account and are always operated with the provider's native credentials again.
"""


@dataclass(frozen=True, repr=False)
class ProviderAccount:
    """One provider account (a CredWeave credential) or the provider's ambient credentials.

    ``credential`` is ``None`` for the ambient account: the provider SDK then resolves its own
    default credentials exactly as it would without InferWeave.
    """

    provider: str
    id: str
    credential: Credential | None = None

    @classmethod
    def ambient(cls, provider: str) -> "ProviderAccount":
        return cls(provider=provider, id=AMBIENT_ACCOUNT_ID)

    @property
    def is_ambient(self) -> bool:
        return self.credential is None

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.credential.metadata if self.credential is not None else {}

    @property
    def secret_fingerprint(self) -> str | None:
        """Nonsecret digest of the current secret values (changes when a secret rotates)."""
        return self.credential.secret_fingerprint if self.credential is not None else None

    @property
    def owner_fingerprint(self) -> str | None:
        """Bind control-plane keys independently of rotatable endpoint proxy credentials.

        A new cloud key is not proof of the same identity. Until a provider offers a verified
        identity migration, existing resources require their original control-plane keys.
        Uses CredWeave's public fingerprint API, without persisting secret material.
        """
        if self.is_ambient:
            return None
        return Credential(
            id=self.id,
            secrets={name: self.secret(name) for name in schema_for(self.provider).required},
        ).secret_fingerprint

    def secret(self, name: str) -> str:
        """Returns a required secret; raises a secret-safe error when it is missing."""
        value = self.optional_secret(name)
        if value is None:
            raise ProviderAuthError(
                f"Account '{self.id}' for provider '{self.provider}' has no '{name}' secret."
            )
        return value

    def optional_secret(self, name: str) -> str | None:
        if self.credential is None:
            return None
        value = self.credential.get_secret(name)
        return value if isinstance(value, str) and value else None

    def has_secret(self, name: str) -> bool:
        return self.optional_secret(name) is not None

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def secret_values(self) -> tuple[str, ...]:
        """All secret values of this account, for redaction and leak guards only."""
        if self.credential is None:
            return ()
        values = (self.credential.get_secret(key) for key in self.credential.secret_keys)
        return tuple(v for v in values if isinstance(v, str) and v)

    def __repr__(self) -> str:
        return f"ProviderAccount(provider={self.provider!r}, id={self.id!r})"

    __str__ = __repr__
