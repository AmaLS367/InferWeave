"""CredWeave credential sources tuned for provider account pools."""

import logging
from collections.abc import Mapping, Sequence

from credweave import (
    Credential,
    CredentialSource,
    CredentialSourceError,
    EnvCredential,
    EnvSource,
)

from inferweave.accounts.schema import CredentialSchema

logger = logging.getLogger(__name__)


class ProviderAccountSource:
    """Merges env-referenced accounts, a hot-reloaded JSON file and custom sources.

    Each env-referenced account is read independently: an account whose variables are unset is
    left out (it receives no new leases and its deployments report it as unavailable) instead of
    breaking every other account of the provider. Every credential is validated against the
    provider's schema; an invalid one is skipped and logged by id only. Duplicate ids across
    sources are a configuration error. Secret values never appear in logs or errors.
    """

    def __init__(
        self,
        schema: CredentialSchema,
        env_accounts: Sequence[EnvCredential] = (),
        sources: Sequence[CredentialSource] = (),
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._schema = schema
        self._env_specs = list(env_accounts)
        # Built lazily: EnvSource reads its variables on construction and raises when unset.
        self._env_sources: dict[str, EnvSource] = {}
        self._environ = environ
        self._sources = list(sources)
        self._reported: set[str] = set()

    @property
    def supports_hot_reload(self) -> bool:
        return True

    def _note(self, key: str, message: str, *args: object) -> None:
        # Warn once per problem so a permanently missing account does not flood the log.
        if key not in self._reported:
            self._reported.add(key)
            logger.warning(message, *args)

    def _merge(self, groups: list[Sequence[Credential]]) -> list[Credential]:
        merged: dict[str, Credential] = {}
        for group in groups:
            for credential in group:
                if credential.id in merged:
                    raise CredentialSourceError(
                        f"{self._schema.provider} account id '{credential.id}' is defined more "
                        "than once."
                    )
                try:
                    self._schema.validate_credential(credential)
                except Exception as err:  # noqa: BLE001 - message is secret-free by design
                    self._note(f"invalid:{credential.id}", "Skipping account: %s", err)
                    continue
                self._reported.discard(f"invalid:{credential.id}")
                merged[credential.id] = credential
        return list(merged.values())

    def _env_group(self) -> list[Credential]:
        found: list[Credential] = []
        for spec in self._env_specs:
            account_id = spec.id
            try:
                source = self._env_sources.get(account_id)
                if source is None:
                    source = EnvSource([spec], environ=self._environ)
                    self._env_sources[account_id] = source
                found.extend(source.get_credentials())
                self._reported.discard(f"env:{account_id}")
            except CredentialSourceError as err:
                # CredWeave names the variable, never its value.
                self._note(
                    f"env:{account_id}",
                    "%s account '%s' is unavailable: %s",
                    self._schema.provider,
                    account_id,
                    err,
                )
        return found

    def get_credentials(self) -> Sequence[Credential]:
        groups: list[Sequence[Credential]] = [self._env_group()]
        groups.extend(source.get_credentials() for source in self._sources)
        return self._merge(groups)

    async def get_credentials_async(self) -> Sequence[Credential]:
        groups: list[Sequence[Credential]] = [self._env_group()]
        for source in self._sources:
            groups.append(await source.get_credentials_async())
        return self._merge(groups)

    def __repr__(self) -> str:
        return (
            f"ProviderAccountSource(provider={self._schema.provider!r}, "
            f"env_accounts={[s.id for s in self._env_specs]!r}, sources={len(self._sources)})"
        )
