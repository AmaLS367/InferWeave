"""Application service orchestrating deployment lifecycle, activity tracking, and autostop."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from inferweave.accounts.config import AccountsConfig
from inferweave.accounts.manager import AccountManager
from inferweave.accounts.models import ProviderAccount
from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import AsyncioWatchdogAdapter
from inferweave.core.exceptions import (
    DeploymentNotFoundError,
    ProviderAuthError,
    ProviderNotFoundError,
    ProviderOperationError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import (
    AutostopAction,
    AutostopPolicy,
    DeploymentLifecycleEvaluator,
    LifecycleState,
)
from inferweave.domain.options import DeploymentOptions
from inferweave.models.deployment import DeploymentStatus
from inferweave.models.enums import DeploymentState, WorkloadType
from inferweave.ports.auth import EndpointAuthPort
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.ports.lifecycle import AutostopWatchdogPort

logger = logging.getLogger(__name__)


class LifecycleService:
    """Orchestrates deployment inactivity monitoring, activity heartbeats, and autostop execution.

    Every remote operation on an existing deployment (status, stop, autostop, reconcile) runs
    under the account recorded on the deployment, resolved through the ``AccountManager``. A
    missing account raises ``AccountUnavailableError``; another account is never substituted.
    """

    def __init__(
        self,
        watchdog_port: AutostopWatchdogPort | None = None,
        repository: DeploymentRepositoryPort | None = None,
        healthcheck_service: Any | None = None,
        provider_resolver: Callable[[str], Any] | None = None,
        endpoint_auth: EndpointAuthPort | None = None,
        activity_persist_interval_seconds: float = 30.0,
        accounts: AccountManager | None = None,
    ) -> None:
        self._accounts = accounts or AccountManager(AccountsConfig.from_env())
        self._watchdog = watchdog_port or AsyncioWatchdogAdapter()
        self._repository: DeploymentRepositoryPort = (
            repository or SqliteDeploymentRepository()
        )
        self._healthcheck_service = healthcheck_service
        self._provider_resolver = provider_resolver
        self._states: dict[str, LifecycleState] = {}
        self._deployments: dict[str, Any] = {}
        self._providers: dict[str, Any] = {}
        self._endpoint_auth = endpoint_auth
        # Persisting `last_activity_at` on every request would turn each inference call into a
        # database write. Writes are therefore throttled; the in-memory timer is always exact.
        self._activity_persist_interval = activity_persist_interval_seconds
        self._last_persisted_activity: dict[str, datetime] = {}

    @property
    def endpoint_auth(self) -> EndpointAuthPort | None:
        """Auth resolver used to authenticate readiness probes made during status refresh."""
        return self._endpoint_auth

    @endpoint_auth.setter
    def endpoint_auth(self, value: EndpointAuthPort | None) -> None:
        self._endpoint_auth = value

    @property
    def accounts(self) -> AccountManager:
        """Account pools used to resolve the owner of every deployment."""
        return self._accounts

    @accounts.setter
    def accounts(self, value: AccountManager) -> None:
        self._accounts = value

    def owner_account(self, record: DeploymentRecord) -> ProviderAccount:
        """The account that owns ``record`` (raises ``AccountUnavailableError`` if removed)."""
        return self._accounts.resolve_owner(record)

    def endpoint_headers(
        self, record: DeploymentRecord, endpoint_url: str | None
    ) -> dict[str, str]:
        """Endpoint auth headers built from the owning account's endpoint credentials."""
        if self._endpoint_auth is None:
            return {}
        return self._endpoint_auth.headers_for(
            record.provider, endpoint_url, self.owner_account(record)
        )

    @property
    def repository(self) -> DeploymentRepositoryPort:
        """Returns the active deployment metadata repository."""
        return self._repository

    async def register_deployment(
        self,
        deployment: Any,
        policy: AutostopPolicy,
        now: datetime | None = None,
        schedule_watchdog: bool = True,
        is_dry_run: bool = False,
        provider: Any | None = None,
        workload_type: WorkloadType | None = None,
        options: DeploymentOptions | None = None,
    ) -> LifecycleState:
        """Registers a freshly created deployment, stores/updates its record, and schedules watchdog.

        If the provider already persisted a record for this deployment (options, timestamps), that
        record is updated in place rather than replaced, so persisted information is preserved.
        """
        current_time = now or datetime.now(UTC)
        state = LifecycleState(
            deployment_id=deployment.id,
            created_at=current_time,
            last_activity_at=current_time,
            policy=policy,
        )
        self._states[deployment.id] = state
        self._deployments[deployment.id] = deployment
        if provider:
            self._providers[deployment.id] = provider

        # Update the provider-persisted record in place, or create one if the provider wrote none
        record = await self._repository.get(deployment.id)
        if record is None:
            record = DeploymentRecord(
                id=deployment.id,
                model=deployment.model,
                provider=deployment.provider,
                state=deployment.state,
                endpoint_url=deployment.endpoint_url,
                created_at=current_time,
                is_dry_run=is_dry_run,
            )
        else:
            record.state = deployment.state
            record.endpoint_url = deployment.endpoint_url or record.endpoint_url
            record.is_dry_run = record.is_dry_run or is_dry_run
        if options is not None:
            record.options = options.model_copy(deep=True)
        # The effective idle policy is persisted so a restarted process can restore the timer.
        record.options.autostop = policy
        record.last_activity_at = current_time
        if workload_type is not None:
            record.workload_type = workload_type
        self._last_persisted_activity[deployment.id] = current_time

        # Synchronous write for fast in-memory availability
        if hasattr(self._repository, "_records"):
            self._repository._records[record.id] = record.model_copy(deep=True)

        await self._repository.save(record)

        if schedule_watchdog:
            await self._schedule_watchdog(deployment.id, policy)

        return state

    async def _schedule_watchdog(
        self, deployment_id: str, policy: AutostopPolicy
    ) -> None:
        """Schedules the periodic destroy-timer check for a policy with an enabled idle limit."""
        if not (
            policy.enabled
            and policy.idle_minutes is not None
            and policy.idle_minutes > 0
        ):
            return
        # Check cadence: sample every 1/4th of the idle duration, capped between 1s and 60s
        interval = min(60.0, max(1.0, float(policy.idle_minutes * 60) / 4.0))

        async def _watchdog_callback() -> None:
            await self.check_and_autostop(deployment_id)

        await self._watchdog.schedule_check(
            deployment_id=deployment_id,
            interval_seconds=interval,
            callback=_watchdog_callback,
        )

    async def rehydrate_deployment(
        self,
        record: DeploymentRecord,
        deployment: Any,
        provider: Any | None = None,
        now: datetime | None = None,
        schedule_watchdog: bool = True,
    ) -> LifecycleState:
        """Re-registers a deployment recovered from its persisted record (e.g. after a restart).

        Unlike ``register_deployment`` this never rewrites identity, options, timestamps or state.
        The idle timer resumes from the persisted ``last_activity_at`` so a process restart does
        not reset a long destroy timer. Legacy records without that field start a fresh timer
        now, and the timestamp is persisted so later restarts no longer reset it.
        """
        current_time = now or datetime.now(UTC)
        last_activity = record.last_activity_at or current_time
        state = LifecycleState(
            deployment_id=record.id,
            created_at=record.created_at,
            last_activity_at=last_activity,
            policy=record.options.autostop,
        )
        self._states[record.id] = state
        self._deployments[record.id] = deployment
        if provider:
            self._providers[record.id] = provider
        self._last_persisted_activity[record.id] = last_activity

        if record.last_activity_at is None:
            record.last_activity_at = last_activity
            await self._repository.save(record)

        if schedule_watchdog:
            await self._schedule_watchdog(record.id, record.options.autostop)
        return state

    async def touch_activity(
        self, deployment_id: str, now: datetime | None = None
    ) -> None:
        """Records real user activity and persists it (throttled) so it survives restarts."""
        current_time = now or datetime.now(UTC)
        self.record_activity(deployment_id, now=current_time)
        last_persisted = self._last_persisted_activity.get(deployment_id)
        if (
            last_persisted is not None
            and 0
            <= (current_time - last_persisted).total_seconds()
            < self._activity_persist_interval
        ):
            return
        record = await self._repository.get(deployment_id)
        if record is None or record.state == DeploymentState.STOPPED:
            return
        record.last_activity_at = current_time
        await self._repository.save(record)
        self._last_persisted_activity[deployment_id] = current_time

    def begin_request(self, deployment_id: str) -> None:
        """Marks a long-running request as in flight so the idle timer cannot expire under it."""
        state = self._states.get(deployment_id)
        if state:
            state.in_flight_requests += 1

    def end_request(self, deployment_id: str) -> None:
        """Marks an in-flight request as finished (never below zero)."""
        state = self._states.get(deployment_id)
        if state and state.in_flight_requests > 0:
            state.in_flight_requests -= 1

    def record_activity(self, deployment_id: str, now: datetime | None = None) -> None:
        """Records client activity or request handling on the specified deployment, resetting idle timer."""
        current_time = now or datetime.now(UTC)
        state = self._states.get(deployment_id)
        if state:
            state.last_activity_at = current_time

    def is_idle(self, deployment_id: str, now: datetime | None = None) -> bool:
        """Checks whether the deployment currently qualifies as idle under its configured policy."""
        state = self._states.get(deployment_id)
        if not state:
            return False
        return DeploymentLifecycleEvaluator.is_idle(state, now)

    async def check_and_autostop(
        self, deployment_id: str, now: datetime | None = None
    ) -> bool:
        """Evaluates idle state and initiates deployment termination if autostop criteria are met."""
        state = self._states.get(deployment_id)
        deployment = self._deployments.get(deployment_id)

        if not state or not deployment or state.is_stopped:
            return False

        if DeploymentLifecycleEvaluator.should_autostop(state, now):
            logger.info(
                "Deployment '%s' exceeded idle limit of %s min(s). Executing autostop action '%s'.",
                deployment_id,
                state.policy.idle_minutes,
                state.policy.action.value,
            )
            await self.stop_deployment(
                deployment_id=deployment_id,
                action=state.policy.action,
                now=now,
            )
            return True

        return False

    async def stop_deployment(
        self,
        deployment_id: str,
        action: AutostopAction | None = None,
        provider: Any | None = None,
        now: datetime | None = None,
    ) -> None:
        """Terminates or pauses a deployment, cancels watchdog monitoring, and persists state."""
        with self._repository.operation_lock(deployment_id) as acquired:
            if not acquired:
                raise ProviderOperationError(
                    "Deployment has an active provisioning or cleanup operation.",
                    FailureKind.TRANSIENT,
                    deployment_id=deployment_id,
                )
            await self._stop_deployment(deployment_id, action, provider, now)

    async def _stop_deployment(
        self,
        deployment_id: str,
        action: AutostopAction | None,
        provider: Any | None,
        now: datetime | None,
    ) -> None:
        current_time = now or datetime.now(UTC)
        state = self._states.get(deployment_id)
        deployment = self._deployments.get(deployment_id)
        record = await self._repository.get(deployment_id)

        if not state and not deployment and not record:
            raise DeploymentNotFoundError(deployment_id)

        target_provider = provider or self._providers.get(deployment_id)
        if not target_provider and record and self._provider_resolver:
            try:
                target_provider = self._provider_resolver(record.provider)
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "Could not resolve provider '%s' for deployment '%s': %s",
                    record.provider,
                    deployment_id,
                    err,
                )

        is_dry_run = (record.is_dry_run if record else False) or (
            getattr(deployment, "is_dry_run", False) if deployment else False
        )

        if not target_provider and not is_dry_run:
            raise ProviderNotFoundError(record.provider if record else "unknown")

        if is_dry_run:
            # For dry-run deployments, stop purely modifies persisted state without invoking cloud provider API
            if state:
                state.is_stopped = True
                state.stopped_at = current_time
            if deployment and hasattr(deployment, "_status"):
                deployment._status.state = DeploymentState.STOPPED
            if record:
                record.mark_stopped(now=current_time)
                await self._repository.save(record)
            await self._watchdog.cancel_check(deployment_id)
            return

        target_action = action or (
            state.policy.action if state else AutostopAction.STOP
        )
        if isinstance(target_action, str):
            target_action = AutostopAction(target_action.lower())
        if record is None:
            # The owning account is only known from the persisted record.
            raise DeploymentNotFoundError(deployment_id)
        if target_provider is None:
            raise ProviderNotFoundError(record.provider)

        try:
            account = self.owner_account(record)
            record.needs_reconciliation = True
            await self._repository.save(record)
            await target_provider.stop(record, account, action=target_action)

            if record.creation_may_continue:
                raise ProviderOperationError(
                    "Delete was requested, but an unacknowledged create may still complete. "
                    "Confirm the create has settled in the provider console before reconciliation.",
                    FailureKind.TRANSIENT,
                    deployment_id=deployment_id,
                )

            # Transition to stopped only upon successful provider stop/down
            if state:
                state.is_stopped = True
                state.stopped_at = current_time
            if deployment and hasattr(deployment, "_status"):
                deployment._status.state = DeploymentState.STOPPED
            record.mark_stopped(now=current_time)
            record.needs_reconciliation = False
            await self._repository.save(record)
        except Exception as err:
            logger.error(
                "Failed to stop deployment '%s' (%s).",
                deployment_id,
                type(err).__name__,
            )
            if isinstance(err, ProviderOperationError | ProviderAuthError):
                raise
            raise ProviderOperationError(
                "Provider stop failed; its outcome is unknown.",
                FailureKind.TRANSIENT,
                deployment_id=deployment_id,
            ) from None
        finally:
            if record.state == DeploymentState.STOPPED:
                await self._watchdog.cancel_check(deployment_id)

    async def refresh_status(
        self,
        deployment_id: str,
        provider: Any | None = None,
        probe: bool = True,
        healthcheck_config: Any | None = None,
    ) -> DeploymentStatus:
        with self._repository.operation_lock(deployment_id) as acquired:
            record = await self._repository.get(deployment_id)
            if record and not record.is_dry_run:
                self.owner_account(record)
            if record and (not acquired or record.creation_may_continue):
                return self._status_from_record(record)
            if not acquired:
                raise ProviderOperationError("Deployment operation is active.", FailureKind.TRANSIENT)
            return await self._refresh_status_locked(deployment_id, provider, probe, healthcheck_config)

    async def _refresh_status_locked(
        self,
        deployment_id: str,
        provider: Any | None = None,
        probe: bool = True,
        healthcheck_config: Any | None = None,
    ) -> DeploymentStatus:
        """Fetches status, executes optional health check probe, and reconciles state."""
        record = await self._repository.get(deployment_id)
        deployment = self._deployments.get(deployment_id)
        state_obj = self._states.get(deployment_id)

        if not record and not deployment and not state_obj:
            raise DeploymentNotFoundError(deployment_id)

        target_provider = provider or self._providers.get(deployment_id)
        if not target_provider and record and self._provider_resolver:
            try:
                target_provider = self._provider_resolver(record.provider)
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "Could not resolve provider '%s' for deployment '%s': %s",
                    record.provider,
                    deployment_id,
                    err,
                )

        # Early exit if deployment is already stopped
        is_already_stopped = (
            (state_obj and state_obj.is_stopped)
            or (record and record.state == DeploymentState.STOPPED)
            or (
                deployment
                and getattr(deployment, "state", None) == DeploymentState.STOPPED
            )
        )
        if is_already_stopped:
            stopped_status = DeploymentStatus(
                id=deployment_id,
                account=record.account
                if record
                else (deployment.account if deployment else None),
                model=record.model
                if record
                else (deployment.model if deployment else "unknown"),
                provider=record.provider
                if record
                else (deployment.provider if deployment else "unknown"),
                state=DeploymentState.STOPPED,
                endpoint_url=record.endpoint_url
                if record
                else (deployment.endpoint_url if deployment else None),
                created_at=record.created_at
                if record
                else (
                    deployment.status.created_at if deployment else datetime.now(UTC)
                ),
                ready_at=record.ready_at if record else None,
            )
            if deployment and hasattr(deployment, "_status"):
                deployment._status = stopped_status
            if record and record.state != DeploymentState.STOPPED:
                record.mark_stopped()
                await self._repository.save(record)
            return stopped_status

        is_dry_run = (record.is_dry_run if record else False) or (
            getattr(deployment, "is_dry_run", False) if deployment else False
        )
        if is_dry_run:
            # For dry-run deployments, return persisted state directly without querying cloud provider API
            dry_status = DeploymentStatus(
                id=deployment_id,
                account=record.account if record else None,
                model=record.model
                if record
                else (deployment.model if deployment else "unknown"),
                provider=record.provider
                if record
                else (deployment.provider if deployment else "unknown"),
                state=record.state if record else DeploymentState.PROVISIONING,
                endpoint_url=record.endpoint_url
                if record
                else (deployment.endpoint_url if deployment else None),
                created_at=record.created_at
                if record
                else (
                    deployment.status.created_at if deployment else datetime.now(UTC)
                ),
                ready_at=record.ready_at if record else None,
            )
            if deployment and hasattr(deployment, "_status"):
                deployment._status = dry_status
            return dry_status

        # 1. Fetch infrastructure status from the provider under the owning account
        if record is None or target_provider is None:
            if deployment is not None and hasattr(deployment, "status"):
                return deployment.status
            raise DeploymentNotFoundError(deployment_id)
        account = self.owner_account(record)
        try:
            resource = await target_provider.status(record, account)
        except ProviderOperationError as err:
            if err.kind not in (FailureKind.TRANSIENT, FailureKind.RATE_LIMIT):
                raise
            # A flaky control plane must not flip a live deployment to a wrong state.
            logger.warning(
                "Could not refresh deployment '%s' (%s); keeping its last known state.",
                deployment_id,
                err.kind.value,
            )
            return self._status_from_record(record)
        raw_status = DeploymentStatus(
            id=deployment_id,
            model=record.model,
            provider=record.provider,
            account=record.account,
            state=resource.state,
            endpoint_url=resource.endpoint_url,
            error_message=record.error_message,
            created_at=record.created_at,
            ready_at=record.ready_at,
        )

        # 2. Preserve known model name rather than accepting 'unknown'
        real_model = (
            record.model
            if (record and raw_status.model == "unknown")
            else (
                raw_status.model
                if raw_status.model != "unknown"
                else (deployment.model if deployment else "unknown")
            )
        )

        # 3. Status reconciliation with active healthcheck probe
        endpoint_url = raw_status.endpoint_url or (
            record.endpoint_url if record else None
        )
        probe_healthy: bool | None = None

        if (
            probe
            and self._healthcheck_service
            and endpoint_url
            and healthcheck_config is not None
            and raw_status.state in {DeploymentState.STARTING, DeploymentState.HEALTHY}
        ):
            try:
                auth_headers = self.endpoint_headers(record, endpoint_url)
                if auth_headers:
                    probe_res = await self._healthcheck_service.check_health(
                        endpoint_url=endpoint_url,
                        config=healthcheck_config,
                        extra_headers=auth_headers,
                    )
                else:
                    probe_res = await self._healthcheck_service.check_health(
                        endpoint_url=endpoint_url,
                        config=healthcheck_config,
                    )
                probe_healthy = probe_res.is_healthy
                if probe_healthy:
                    self.record_activity(deployment_id)
            except Exception as err:  # noqa: BLE001
                logger.debug("Health probe during refresh failed: %s", err)
                probe_healthy = False

        reconciled_state = DeploymentLifecycleEvaluator.reconcile_state(
            infra_state=raw_status.state,
            probe_is_healthy=probe_healthy,
            is_stopped=(record.state == DeploymentState.STOPPED if record else False),
            current_state=(record.state if record else None),
        )

        if record.creation_may_continue:
            reconciled_state = record.state

        updated_status = DeploymentStatus(
            id=deployment_id,
            model=real_model,
            provider=raw_status.provider,
            account=record.account,
            state=reconciled_state,
            endpoint_url=endpoint_url,
            error_message=raw_status.error_message
            or (record.error_message if record else None),
            created_at=raw_status.created_at,
            ready_at=raw_status.ready_at or (record.ready_at if record else None),
        )

        # 4. Persist updated status to repository and deployment handle
        if resource.state == DeploymentState.STOPPED and not record.creation_may_continue:
            record.needs_reconciliation = False
        if record:
            if (
                reconciled_state == DeploymentState.HEALTHY
                and record.state != DeploymentState.HEALTHY
            ):
                record.mark_healthy(endpoint_url=endpoint_url)
                updated_status.ready_at = updated_status.ready_at or record.ready_at
            record.state = reconciled_state
            record.endpoint_url = endpoint_url
            await self._repository.save(record)

        if deployment and hasattr(deployment, "_status"):
            deployment._status = updated_status

        return updated_status

    @staticmethod
    def _status_from_record(record: DeploymentRecord) -> DeploymentStatus:
        return DeploymentStatus(
            id=record.id,
            model=record.model,
            provider=record.provider,
            account=record.account,
            state=record.state,
            endpoint_url=record.endpoint_url,
            error_message=record.error_message,
            created_at=record.created_at,
            ready_at=record.ready_at,
        )

    async def reconcile(
        self,
        min_age_seconds: float = 1800.0,
        now: datetime | None = None,
        *,
        confirmed_settled: tuple[str, ...] = (),
    ) -> list[DeploymentRecord]:
        """Resolves deployments whose remote state is unknown, under their owning accounts.

        * A failed/cancelled provisioning whose resource still exists is destroyed.
        * A resource confirmed absent clears the flag.
        * Records still marked PROVISIONING are only touched once older than
          ``min_age_seconds``, so a deploy still running in another process is left alone.

        Accounts that are no longer configured are skipped (logged) and stay flagged.
        Returns the records that were updated.
        """
        current_time = now or datetime.now(UTC)
        updated: list[DeploymentRecord] = []
        for snapshot in await self._repository.list_all():
            with self._repository.operation_lock(snapshot.id) as acquired:
                if not acquired:
                    continue
                record = await self._repository.get(snapshot.id)
                if record is None:
                    continue
                if record.id in confirmed_settled:
                    # Explicit operator confirmation, after checking the provider's operation
                    # history. Never inferred from age or a momentary absence snapshot.
                    self.owner_account(record)
                    record.creation_may_continue = False
                    await self._repository.save(record)
                result = await self._reconcile_record(
                    record, min_age_seconds, current_time
                )
                if result:
                    updated.append(record)
        return updated

    async def _reconcile_record(
        self,
        record: DeploymentRecord,
        min_age_seconds: float,
        current_time: datetime,
    ) -> bool:
        if not record.needs_reconciliation or record.is_dry_run:
            return False
        in_flight = record.state in {
            DeploymentState.PENDING,
            DeploymentState.PROVISIONING,
        }
        if (
            in_flight
            and (current_time - record.created_at).total_seconds() < min_age_seconds
        ):
            return False
        provider = (
            self._provider_resolver(record.provider)
            if self._provider_resolver
            else None
        )
        if provider is None:
            return False
        if record.creation_may_continue:
            return False
        try:
            account = self.owner_account(record)
            if await provider.resource_exists(record, account):
                await provider.stop(record, account, action=AutostopAction.DOWN)
                if await provider.resource_exists(record, account):
                    return False
                record.mark_stopped(now=current_time)
            elif record.is_active():
                record.mark_failed("Provisioning did not complete; no resource exists.")
        except Exception as err:  # noqa: BLE001 - keep reconciling other deployments
            logger.warning(
                "Could not reconcile deployment '%s' (%s); it stays flagged.",
                record.id,
                type(err).__name__,
            )
            return False
        record.needs_reconciliation = False
        await self._repository.save(record)
        return True

    def get_state(self, deployment_id: str) -> LifecycleState | None:
        """Returns the current LifecycleState for the given deployment ID."""
        return self._states.get(deployment_id)

    async def get_record(self, deployment_id: str) -> DeploymentRecord | None:
        """Returns the persisted DeploymentRecord from the repository."""
        return await self._repository.get(deployment_id)

    async def list_records(self) -> list[DeploymentRecord]:
        """Returns all persisted DeploymentRecords from the repository."""
        return await self._repository.list_all()

    async def unregister_deployment(self, deployment_id: str) -> None:
        """Unregisters deployment from lifecycle monitoring and cancels background watchdog tasks."""
        self._states.pop(deployment_id, None)
        self._deployments.pop(deployment_id, None)
        self._last_persisted_activity.pop(deployment_id, None)
        await self._watchdog.cancel_check(deployment_id)

    async def close(self) -> None:
        """Releases watchdog resources and cancels all scheduled checks."""
        await self._watchdog.close()
