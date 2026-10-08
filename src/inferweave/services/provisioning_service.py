"""Provisioning use case: account scheduling, safe failover and duplicate-free recovery.

For each attempt the service:

1. leases an account from the provider's CredWeave pool (or the ambient account);
2. writes the deployment record *before* the create call, with the owning account, the remote
   resource name and ``needs_reconciliation=True`` (write-ahead), so a crash or timeout can
   always be reconciled under the right account;
3. calls the provider and classifies any failure;
4. if the failure is not a definitive rejection, looks the resource up under the same account
   and destroys it before anything else is tried, so a paid resource is never duplicated;
   if that cannot be established, it stops immediately with ``ProvisioningUncertainError``;
5. reports the outcome to CredWeave and fails over to the next account/provider only when the
   failure kind allows it, within a fixed attempt budget.
"""

import asyncio
import contextlib
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from inferweave.accounts.manager import AccountLease, AccountManager
from inferweave.accounts.models import ProviderAccount
from inferweave.core.exceptions import (
    InferWeaveError,
    NoAccountAvailableError,
    ProviderAuthError,
    ProviderOperationError,
    ProvisioningUncertainError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import AutostopAction
from inferweave.domain.options import DeploymentOptions
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState
from inferweave.models.profile import ModelProfile
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.providers.base import ComputeProvider
from inferweave.runtimes.base import RuntimeSpec

logger = logging.getLogger(__name__)

PLATFORM_CREDENTIAL_ENV = frozenset(
    {
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
        "LIGHTNING_USER_ID",
        "LIGHTNING_API_KEY",
        "LIGHTNING_AUTH_TOKEN",
        "RUNPOD_API_KEY",
        "VAST_API_KEY",
    }
)
"""Control-plane credential variables that must never reach a model container."""


@dataclass(frozen=True)
class ProvisionCandidate:
    """One provider to try, with the request/runtime rendered for it."""

    provider: ComputeProvider
    request: DeploymentRequest
    runtime: RuntimeSpec


def _contains(value: Any, secrets: tuple[str, ...]) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).upper() in PLATFORM_CREDENTIAL_ENV or _contains(item, secrets)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set):
        return any(_contains(item, secrets) for item in value)
    return isinstance(value, str) and any(secret in value for secret in secrets)


def guard_runtime(
    runtime: RuntimeSpec, options: DeploymentOptions | None, secrets: Iterable[str]
) -> None:
    """Refuses to ship platform credentials into a model container or persisted options."""
    known = tuple(s for s in secrets if s and len(s) >= 4)
    ambient = tuple(
        v for k in PLATFORM_CREDENTIAL_ENV if len(v := os.environ.get(k, "")) >= 4
    )
    if _contains(runtime.model_dump(), known + ambient) or (
        options is not None and _contains(options.model_dump(), known + ambient)
    ):
        raise ProviderAuthError(
            "Provider platform credentials must not be injected into model runtimes, "
            "environment variables or persisted deployment options."
        )


class ProvisioningService:
    """Creates deployments under scheduled accounts with bounded, duplicate-free failover."""

    def __init__(
        self, accounts: AccountManager, repository: DeploymentRepositoryPort
    ) -> None:
        self._accounts = accounts
        self._repository = repository

    @property
    def accounts(self) -> AccountManager:
        return self._accounts

    async def provision(
        self,
        candidates: list[ProvisionCandidate],
        profile: ModelProfile,
        *,
        account: str | None = None,
    ) -> tuple[DeploymentRecord, ComputeProvider]:
        """Provisions on the first candidate/account that succeeds.

        Args:
            candidates: Providers in preference order (more than one only for ``auto``).
            account: Pin one account id (no failover to other accounts or providers).
        """
        if not candidates:
            raise NoAccountAvailableError("No provider candidates to provision on.")
        first = candidates[0]
        if first.request.dry_run:
            return await self._dry_run(first, profile), first.provider
        if account is not None:
            candidates = candidates[:1]

        attempts: list[tuple[str, str, str]] = []
        last_error: Exception | None = None
        retry_after: float | None = None
        budget = self._accounts.max_attempts
        for candidate in candidates:
            provider = candidate.provider
            tried: set[str] = set()
            while len(attempts) < budget:
                try:
                    lease = await self._accounts.acquire(
                        provider.name, exclude=tried, pin=account
                    )
                except NoAccountAvailableError as err:
                    if err.retry_after is not None:
                        retry_after = (
                            err.retry_after
                            if retry_after is None
                            else min(retry_after, err.retry_after)
                        )
                    break
                tried.add(lease.account_id)
                try:
                    record = await self._attempt(candidate, profile, lease)
                except ProviderOperationError as err:
                    attempts.append((provider.name, lease.account_id, err.kind.value))
                    last_error = err
                    if not err.kind.allows_failover or account is not None:
                        raise
                    logger.warning(
                        "Provisioning on %s account '%s' failed (%s); trying the next account.",
                        provider.name,
                        lease.account_id,
                        err.kind.value,
                    )
                    continue
                return record, provider
            if len(attempts) >= budget:
                break

        if len(attempts) == 1 and last_error is not None:
            raise last_error
        summary = (
            ", ".join(f"{p}/{a}: {k}" for p, a, k in attempts)
            or "no account was available"
        )
        raise NoAccountAvailableError(
            f"Could not provision '{profile.id}' after {len(attempts)} attempt(s) ({summary}).",
            attempts=attempts,
            retry_after=retry_after,
        ) from last_error

    async def _dry_run(
        self, candidate: ProvisionCandidate, profile: ModelProfile
    ) -> DeploymentRecord:
        provider, request, runtime = (
            candidate.provider,
            candidate.request,
            candidate.runtime,
        )
        placeholder = ProviderAccount.ambient(provider.name)
        provider.preflight(request, profile, runtime, placeholder)
        deployment_id = provider.new_deployment_id(profile)
        record = DeploymentRecord(
            id=deployment_id,
            model=profile.id,
            provider=provider.name,
            account=None,
            resource=provider.resource_ref(deployment_id, request, placeholder),
            state=DeploymentState.PROVISIONING,
            endpoint_url=provider.dry_run_endpoint(deployment_id, runtime),
            options=(request.options or DeploymentOptions()).model_copy(deep=True),
            is_dry_run=True,
            workload_type=profile.workload_type,
        )
        await self._repository.save(record)
        return record

    async def _attempt(
        self, candidate: ProvisionCandidate, profile: ModelProfile, lease: AccountLease
    ) -> DeploymentRecord:
        provider, request, runtime = (
            candidate.provider,
            candidate.request,
            candidate.runtime,
        )
        account = lease.account
        try:
            guard_runtime(runtime, request.options, self._accounts.secret_values())
            provider.preflight(request, profile, runtime, account)
            deployment_id = provider.new_deployment_id(profile)
            resource = provider.resource_ref(deployment_id, request, account)
        except BaseException:
            # Local validation failed before anything remote happened; not the account's fault.
            await lease.report_success()
            raise

        options = (request.options or DeploymentOptions()).model_copy(deep=True)
        if provider.cleanup_failed_deployment:
            options.cleanup_on_failure = True
        record = DeploymentRecord(
            id=deployment_id,
            model=profile.id,
            provider=provider.name,
            account=account.id,
            resource=resource,
            state=DeploymentState.PROVISIONING,
            options=options,
            workload_type=profile.workload_type,
            needs_reconciliation=True,
            owner_fingerprint=account.owner_fingerprint,
            creation_may_continue=True,
        )
        with self._repository.operation_lock(record.id) as acquired:
            if not acquired:
                await lease.report_success()
                raise ProviderOperationError(
                    "Deployment operation is already active.",
                    FailureKind.INVALID_REQUEST,
                    resource_may_exist=False,
                    deployment_id=record.id,
                )
            try:
                await self._repository.save(
                    record
                )  # owner + name before the create call
                try:
                    await provider.prepare(record, account)
                except ProviderOperationError as err:
                    err.resource_may_exist = False
                    err.operation_may_continue = False
                    await self._settle_failure(provider, record, lease, err)
                    raise
                except Exception as err:  # noqa: BLE001 - read-only preparation, withhold SDK text
                    wrapped = ProviderOperationError(
                        f"{provider.name} preparation failed ({type(err).__name__}).",
                        FailureKind.TRANSIENT, resource_may_exist=False, deployment_id=record.id,
                    )
                    await self._settle_failure(provider, record, lease, wrapped)
                    raise wrapped from None
                await self._repository.save(
                    record
                )  # durable ownership before any creation
            except BaseException:
                await lease.report_success()
                raise
            try:
                return await self._create(candidate, profile, lease, record)
            finally:
                await asyncio.shield(lease.report_success())

    async def _create(
        self,
        candidate: ProvisionCandidate,
        profile: ModelProfile,
        lease: AccountLease,
        record: DeploymentRecord,
    ) -> DeploymentRecord:
        provider, request, runtime = (
            candidate.provider,
            candidate.request,
            candidate.runtime,
        )
        account = lease.account

        try:
            result = await provider.provision(
                record, request, profile, runtime, account
            )
            if _contains(result.endpoint_url, account.secret_values()):
                record.creation_may_continue = False
                raise ProviderOperationError(
                    "Provider returned credentials in the endpoint URL.",
                    FailureKind.INVALID_REQUEST,
                    resource_may_exist=True,
                    deployment_id=record.id,
                )
        except ProviderOperationError as err:
            await self._settle_failure(provider, record, lease, err)
            raise
        except InferWeaveError as err:
            # Our own sanitized errors (configuration, platform...): clean up, never fail over.
            wrapped = ProviderOperationError(
                f"{provider.name} provisioning failed ({type(err).__name__}).",
                FailureKind.INVALID_REQUEST,
                resource_may_exist=True,
                deployment_id=record.id,
                operation_may_continue=record.creation_may_continue,
            )
            await self._settle_failure(provider, record, lease, wrapped)
            raise
        except Exception as err:  # noqa: BLE001 - SDK error text may contain credentials
            wrapped = ProviderOperationError(
                f"{provider.name} provisioning failed ({type(err).__name__}).",
                FailureKind.TRANSIENT,
                resource_may_exist=True,
                deployment_id=record.id,
                operation_may_continue=True,
            )
            await self._settle_failure(provider, record, lease, wrapped)
            raise wrapped from None
        except BaseException as err:
            # Cancelled or interrupted mid-create: try to leave nothing billed behind.
            record.creation_may_continue = getattr(
                err, "operation_may_continue", record.creation_may_continue
            )
            await asyncio.shield(self._cleanup_after_abort(provider, record, account))
            await lease.report_success()
            raise

        record.state = result.state
        record.endpoint_url = result.endpoint_url
        if record.resource is not None and result.resource_id:
            record.resource.resource_id = result.resource_id
        record.needs_reconciliation = False
        record.creation_may_continue = False
        await self._repository.save(record)
        await lease.report_success()
        return record

    async def _settle_failure(
        self,
        provider: ComputeProvider,
        record: DeploymentRecord,
        lease: AccountLease,
        err: ProviderOperationError,
    ) -> None:
        """Reconciles remote state, persists the failure and reports the account outcome."""
        account = lease.account
        record.creation_may_continue = err.operation_may_continue
        if err.resource_may_exist:
            clean = await self._remove_if_present(provider, record, account)
            if not clean or record.creation_may_continue:
                record.mark_failed(
                    "Provisioning outcome unknown; a billed resource may exist. Run reconcile() "
                    "or stop() for this deployment."
                )
                record.needs_reconciliation = True
                await self._repository.save(record)
                await lease.report_failure(err)
                raise ProvisioningUncertainError(
                    record.id, provider.name, account.id
                ) from None
        record.needs_reconciliation = False
        record.mark_failed(
            f"Provisioning failed on account '{account.id}' ({err.kind.value})."
        )
        await self._repository.save(record)
        await lease.report_failure(err)

    async def _remove_if_present(
        self,
        provider: ComputeProvider,
        record: DeploymentRecord,
        account: ProviderAccount,
    ) -> bool:
        """True when the resource is confirmed absent (or was removed) under ``account``."""
        try:
            # A timed-out create can still be running in a cloud API/server after the worker
            # exited. A snapshot of absence (or teardown) cannot establish quiescence.
            if record.creation_may_continue:
                return False
            if not await provider.resource_exists(record, account):
                return True
            logger.warning(
                "Deployment '%s' exists on %s account '%s' after a failed create; removing it "
                "before any retry.",
                record.id,
                provider.name,
                account.id,
            )
            await provider.stop(record, account, action=AutostopAction.DOWN)
            return not await provider.resource_exists(record, account)
        except Exception as err:  # noqa: BLE001 - reconciliation failure means "unknown"
            logger.error(
                "Could not reconcile deployment '%s' on %s account '%s' (%s).",
                record.id,
                provider.name,
                account.id,
                type(err).__name__,
            )
            return False

    async def _cleanup_after_abort(
        self,
        provider: ComputeProvider,
        record: DeploymentRecord,
        account: ProviderAccount,
    ) -> None:
        with contextlib.suppress(Exception):
            if await self._remove_if_present(provider, record, account):
                record.needs_reconciliation = False
                record.mark_failed("Provisioning was cancelled; no resource remains.")
            else:
                record.mark_failed(
                    "Provisioning was cancelled; run reconcile() or stop()."
                )
            await self._repository.save(record)
