"""CredWeave-backed account pools: scheduling, outcome reporting and owner resolution."""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

from credweave import (
    Clock,
    CredentialPool,
    CredentialSource,
    CredentialState,
    CredWeaveError,
    EnvCredential,
    JsonSource,
    Lease,
    NoCredentialsAvailableError,
    Outcome,
)

from inferweave.accounts.config import AccountsConfig, ProviderAccounts
from inferweave.accounts.models import AMBIENT_ACCOUNT_ID, ProviderAccount
from inferweave.accounts.schema import schema_for
from inferweave.accounts.sources import ProviderAccountSource
from inferweave.accounts.strategy import (
    ScopedSelectionContext,
    ScopedStrategy,
    make_strategy,
)
from inferweave.core.exceptions import (
    AccountConfigurationError,
    AccountUnavailableError,
    NoAccountAvailableError,
    ProviderOperationError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord

logger = logging.getLogger(__name__)

REDACTED = "***"


@dataclass(frozen=True)
class AccountHealth:
    """Nonsecret scheduling state of one pooled account."""

    provider: str
    account_id: str
    state: str
    in_flight: int
    consecutive_failures: int
    total_leases: int
    cooldown_until: datetime | None


class AccountLease:
    """An account selected for one provisioning attempt; report exactly one outcome."""

    def __init__(
        self,
        manager: "AccountManager",
        account: ProviderAccount,
        lease: Lease | None,
    ) -> None:
        self._manager = manager
        self.account = account
        self._lease = lease
        self._reported = False

    @property
    def account_id(self) -> str:
        return self.account.id

    async def report_success(self) -> None:
        await self._report(Outcome.success())

    async def report_failure(self, error: ProviderOperationError) -> None:
        await self._report(self._manager.outcome_for(self.account.provider, error))

    async def _report(self, outcome: Outcome) -> None:
        if self._reported or self._lease is None:
            self._reported = True
            return
        self._reported = True
        pool = self._manager.pool(self.account.provider)
        try:
            await pool.report(self._lease, outcome)
        except CredWeaveError as err:
            # A lost report must never mask the provisioning result itself.
            logger.warning(
                "Could not report outcome for %s account '%s': %s",
                self.account.provider,
                self.account.id,
                type(err).__name__,
            )

    def __repr__(self) -> str:
        return f"AccountLease(account={self.account!r})"


class _Pool:
    def __init__(self, provider: str, settings: ProviderAccounts, pool: CredentialPool) -> None:
        self.provider = provider
        self.settings = settings
        self.pool = pool


class AccountManager:
    """Owns one CredWeave ``CredentialPool`` per provider with configured accounts.

    * New deployments ``acquire`` an account through the provider's selection strategy and
      ``report`` the outcome, which drives CredWeave cooldowns, health and failover.
    * Existing deployments never go through scheduling: ``resolve_owner`` returns exactly the account
      and control-credential generation recorded on the deployment or raises :class:`AccountUnavailableError`.
    """

    def __init__(
        self,
        config: AccountsConfig | None = None,
        *,
        clock: Clock | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or AccountsConfig()
        self._pools: dict[str, _Pool] = {}
        for provider, settings in self.config.providers.items():
            schema = schema_for(provider)
            sources: list[CredentialSource] = list(settings.sources)
            if settings.credentials_file is not None:
                sources.append(JsonSource(settings.credentials_file))
            source = ProviderAccountSource(
                schema,
                env_accounts=[
                    EnvCredential(
                        id=spec.id,
                        secrets={
                            name: var for name, var in spec.env.items() if name in schema.required
                        },
                        optional_secrets={
                            name: var
                            for name, var in spec.env.items()
                            if name not in schema.required
                        },
                        metadata=spec.credweave_metadata(),
                    )
                    for spec in settings.accounts
                ],
                sources=sources,
                environ=environ,
            )
            try:
                pool = CredentialPool(
                    source=source,
                    strategy=ScopedStrategy(make_strategy(settings.strategy)),
                    clock=clock,
                    default_cooldown=settings.cooldown_seconds,
                    max_consecutive_failures=settings.max_consecutive_failures,
                )
            except CredWeaveError as err:
                # CredWeave errors name files/variables, never secret values.
                raise AccountConfigurationError(f"{provider} accounts: {err}") from None
            self._pools[provider] = _Pool(provider, settings, pool)

    def __repr__(self) -> str:
        return f"AccountManager(pools={sorted(self._pools)})"

    @property
    def max_attempts(self) -> int:
        return self.config.max_attempts

    def has_pool(self, provider: str) -> bool:
        return provider.lower() in self._pools

    def pool(self, provider: str) -> CredentialPool:
        return self._pools[provider.lower()].pool

    def account_ids(self, provider: str) -> list[str]:
        """Ids of the accounts currently provided for ``provider`` (ambient when unpooled)."""
        entry = self._pools.get(provider.lower())
        if entry is None:
            return [AMBIENT_ACCOUNT_ID]
        return [c.id for c in entry.pool.source.get_credentials()]

    async def acquire(
        self,
        provider: str,
        *,
        exclude: Iterable[str] = (),
        pin: str | None = None,
    ) -> AccountLease:
        """Leases the next eligible account of ``provider`` for one provisioning attempt."""
        name = provider.lower()
        entry = self._pools.get(name)
        excluded = frozenset(exclude)
        if entry is None:
            if pin not in (None, AMBIENT_ACCOUNT_ID):
                raise AccountUnavailableError(name, str(pin))
            if AMBIENT_ACCOUNT_ID in excluded:
                raise NoAccountAvailableError(
                    f"No remaining account for provider '{name}' (ambient credentials failed)."
                )
            return AccountLease(self, ProviderAccount.ambient(name), None)
        if pin is not None and pin not in self.account_ids(name):
            raise AccountUnavailableError(name, pin)
        try:
            lease = await entry.pool.acquire(ScopedSelectionContext(exclude=excluded, pin=pin))
        except NoCredentialsAvailableError:
            raise NoAccountAvailableError(
                f"No eligible {name} account is available (all accounts are cooling down, "
                "rate limited, revoked, at capacity or already tried).",
                retry_after=self._earliest_recovery(entry),
            ) from None
        return AccountLease(self, ProviderAccount(name, lease.credential_id, lease.credential), lease)

    def resolve(
        self, provider: str, account_id: str | None, deployment_id: str | None = None
    ) -> ProviderAccount:
        """Returns the exact account that owns a deployment; never substitutes another one."""
        name = provider.lower()
        if account_id is None or account_id == AMBIENT_ACCOUNT_ID:
            return ProviderAccount.ambient(name)
        entry = self._pools.get(name)
        credential = entry.pool.get_credential(account_id) if entry is not None else None
        if credential is None:
            raise AccountUnavailableError(name, account_id, deployment_id)
        state = entry.pool.get_record(account_id) if entry is not None else None
        if state is not None and state.state is CredentialState.REVOKED:
            raise AccountUnavailableError(name, account_id, deployment_id)
        return ProviderAccount(name, credential.id, credential)

    def resolve_owner(self, record: DeploymentRecord) -> ProviderAccount:
        account = self.resolve(record.provider, record.account, record.id)
        if record.owner_fingerprint is not None and account.owner_fingerprint != record.owner_fingerprint:
            raise AccountUnavailableError(record.provider, str(record.account), record.id)
        return account

    def outcome_for(self, provider: str, error: ProviderOperationError) -> Outcome:
        """Maps a classified provider failure to the CredWeave outcome for its account.

        Only confirmed invalid credentials revoke an account. Throttling, quota and permission
        problems park it without touching its health, and GPU stock-outs or invalid requests
        are not the credential's fault at all (reported as a successful use of the credential).
        """
        settings = self._pools[provider.lower()].settings if self.has_pool(provider) else None
        meta = {"inferweave_failure": error.kind.value}
        if error.status_code is not None:
            meta["status_code"] = str(error.status_code)
        kind = error.kind
        if kind is FailureKind.AUTH:
            return Outcome.auth_failed(reason="credentials rejected", metadata=meta)
        if kind is FailureKind.RATE_LIMIT:
            return Outcome.rate_limited(retry_after=error.retry_after, metadata=meta)
        if kind is FailureKind.QUOTA:
            return Outcome.quota_exhausted(
                retry_after=error.retry_after
                if error.retry_after is not None
                else (settings.quota_cooldown_seconds if settings else None),
                metadata=meta,
            )
        if kind is FailureKind.PERMISSION:
            return Outcome.rate_limited(
                retry_after=settings.permission_cooldown_seconds if settings else None,
                reason="permission denied",
                metadata=meta,
            )
        if kind is FailureKind.TRANSIENT:
            return Outcome.transient_error(retry_after=error.retry_after, metadata=meta)
        # CAPACITY / INVALID_REQUEST: the credential itself worked.
        return Outcome.success(metadata=meta)

    def secret_values(self) -> tuple[str, ...]:
        """Every secret value of every configured account (for leak guards and redaction)."""
        values: list[str] = []
        for entry in self._pools.values():
            for credential in entry.pool.source.get_credentials():
                account = ProviderAccount(entry.provider, credential.id, credential)
                values.extend(account.secret_values())
        return tuple(values)

    def redact(self, text: str, extra: Iterable[str] = ()) -> str:
        """Replaces any known secret value in ``text`` with ``***``."""
        for value in sorted({*self.secret_values(), *extra}, key=len, reverse=True):
            if value:
                text = text.replace(value, REDACTED)
        return text

    def health(self) -> list[AccountHealth]:
        """Nonsecret state of every pooled account (unpooled providers use ambient creds)."""
        report: list[AccountHealth] = []
        for entry in self._pools.values():
            records = {r.credential_id: r for r in entry.pool.list_records()}
            for credential in entry.pool.source.get_credentials():
                record = records.get(credential.id)
                report.append(
                    AccountHealth(
                        provider=entry.provider,
                        account_id=credential.id,
                        state=str(record.state) if record else "available",
                        in_flight=record.in_flight_leases if record else 0,
                        consecutive_failures=record.consecutive_failures if record else 0,
                        total_leases=record.total_leases if record else 0,
                        cooldown_until=record.cooldown_until if record else None,
                    )
                )
        return report

    async def reset(self, provider: str, account_id: str) -> None:
        """Clears cooldown/health state, e.g. after a quota top-up (CredWeave reset)."""
        await self.pool(provider).reset_credential_async(account_id)

    async def authorize(self, provider: str, account_id: str) -> None:
        """Re-activates a revoked account after its secret was repaired (CredWeave authorize)."""
        await self.pool(provider).authorize_secret_async(account_id)

    def _earliest_recovery(self, entry: _Pool) -> float | None:
        now = entry.pool.clock.now()
        delays = [
            (r.cooldown_until - now).total_seconds()
            for r in entry.pool.list_records()
            if r.cooldown_until is not None and r.cooldown_until > now
        ]
        return max(0.0, min(delays)) if delays else None
