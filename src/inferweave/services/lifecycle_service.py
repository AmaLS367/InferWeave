"""Application service orchestrating deployment lifecycle, activity tracking, and autostop."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import AsyncioWatchdogAdapter
from inferweave.core.exceptions import DeploymentNotFoundError, ProviderNotFoundError
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.lifecycle import (
    AutostopAction,
    AutostopPolicy,
    DeploymentLifecycleEvaluator,
    LifecycleState,
)
from inferweave.models.deployment import DeploymentStatus
from inferweave.models.enums import DeploymentState
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.ports.lifecycle import AutostopWatchdogPort

logger = logging.getLogger(__name__)


class LifecycleService:
    """Orchestrates deployment inactivity monitoring, activity heartbeats, and autostop execution."""

    def __init__(
        self,
        watchdog_port: AutostopWatchdogPort | None = None,
        repository: DeploymentRepositoryPort | None = None,
        healthcheck_service: Any | None = None,
        provider_resolver: Callable[[str], Any] | None = None,
    ) -> None:
        self._watchdog = watchdog_port or AsyncioWatchdogAdapter()
        self._repository: DeploymentRepositoryPort = (
            repository or SqliteDeploymentRepository()
        )
        self._healthcheck_service = healthcheck_service
        self._provider_resolver = provider_resolver
        self._states: dict[str, LifecycleState] = {}
        self._deployments: dict[str, Any] = {}
        self._providers: dict[str, Any] = {}

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
    ) -> LifecycleState:
        """Registers a deployment for lifecycle tracking, stores its record, and schedules watchdog."""
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

        # Schedule or sync record in repository
        record = DeploymentRecord(
            id=deployment.id,
            model=deployment.model,
            provider=deployment.provider,
            state=deployment.state,
            endpoint_url=deployment.endpoint_url,
            created_at=current_time,
            is_dry_run=is_dry_run,
        )

        # Synchronous write for fast in-memory availability
        if hasattr(self._repository, "_records"):
            self._repository._records[record.id] = record.model_copy(deep=True)

        await self._repository.save(record)

        if (
            schedule_watchdog
            and policy.enabled
            and policy.idle_minutes is not None
            and policy.idle_minutes > 0
        ):
            # Check cadence: sample every 1/4th of the idle duration, capped between 1s and 60s
            interval = min(60.0, max(1.0, float(policy.idle_minutes * 60) / 4.0))

            async def _watchdog_callback() -> None:
                await self.check_and_autostop(deployment.id)

            await self._watchdog.schedule_check(
                deployment_id=deployment.id,
                interval_seconds=interval,
                callback=_watchdog_callback,
            )

        return state

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

        target_action = action or (
            state.policy.action if state else AutostopAction.STOP
        )
        if isinstance(target_action, str):
            target_action = AutostopAction(target_action.lower())

        try:
            if target_provider and hasattr(target_provider, "stop"):
                await target_provider.stop(deployment_id, action=target_action)

            # Transition to stopped only upon successful provider stop/down
            if state:
                state.is_stopped = True
                state.stopped_at = current_time
            if deployment and hasattr(deployment, "_status"):
                deployment._status.state = DeploymentState.STOPPED
            if record:
                record.mark_stopped(now=current_time)
                await self._repository.save(record)
        except Exception as err:
            logger.error("Failed to stop deployment '%s': %s", deployment_id, err)
            raise
        finally:
            await self._watchdog.cancel_check(deployment_id)

    async def refresh_status(
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

        # 1. Fetch raw status from provider or deployment handle
        raw_status: DeploymentStatus | None = None
        if target_provider and hasattr(target_provider, "get_status"):
            raw_status = await target_provider.get_status(deployment_id)
        elif (
            deployment and hasattr(deployment, "_refresh_fn") and deployment._refresh_fn
        ):
            raw_status = await deployment._refresh_fn()
        elif deployment and hasattr(deployment, "status"):
            raw_status = deployment.status

        # If unable to fetch, construct fallback status
        if not raw_status:
            model = record.model if record else "unknown"
            prov_name = record.provider if record else "unknown"
            st = record.state if record else DeploymentState.PENDING
            ep = record.endpoint_url if record else None
            return DeploymentStatus(
                id=deployment_id,
                model=model,
                provider=prov_name,
                state=st,
                endpoint_url=ep,
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
            and raw_status.state == DeploymentState.HEALTHY
        ):
            try:
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

        updated_status = DeploymentStatus(
            id=deployment_id,
            model=real_model,
            provider=raw_status.provider,
            state=reconciled_state,
            endpoint_url=endpoint_url,
            error_message=raw_status.error_message
            or (record.error_message if record else None),
            created_at=raw_status.created_at,
            ready_at=raw_status.ready_at or (record.ready_at if record else None),
        )

        # 4. Persist updated status to repository and deployment handle
        if record:
            record.state = reconciled_state
            record.endpoint_url = endpoint_url
            await self._repository.save(record)

        if deployment and hasattr(deployment, "_status"):
            deployment._status = updated_status

        return updated_status

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
        await self._watchdog.cancel_check(deployment_id)

    async def close(self) -> None:
        """Releases watchdog resources and cancels all scheduled checks."""
        await self._watchdog.close()
