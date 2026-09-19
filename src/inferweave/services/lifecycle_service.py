"""Application service orchestrating deployment lifecycle, activity tracking, and autostop."""

import logging
from datetime import UTC, datetime
from typing import Any

from inferweave.adapters.lifecycle.watchdog import AsyncioWatchdogAdapter
from inferweave.domain.lifecycle import (
    AutostopPolicy,
    DeploymentLifecycleEvaluator,
    LifecycleState,
)
from inferweave.ports.lifecycle import AutostopWatchdogPort

logger = logging.getLogger(__name__)


class LifecycleService:
    """Orchestrates deployment inactivity monitoring, activity heartbeats, and autostop execution."""

    def __init__(self, watchdog_port: AutostopWatchdogPort | None = None) -> None:
        self._watchdog = watchdog_port or AsyncioWatchdogAdapter()
        self._states: dict[str, LifecycleState] = {}
        self._deployments: dict[str, Any] = {}

    def register_deployment(
        self,
        deployment: Any,
        policy: AutostopPolicy,
        now: datetime | None = None,
        schedule_watchdog: bool = True,
    ) -> LifecycleState:
        """Registers a deployment for lifecycle tracking and optionally schedules a watchdog check."""
        current_time = now or datetime.now(UTC)
        state = LifecycleState(
            deployment_id=deployment.id,
            created_at=current_time,
            last_activity_at=current_time,
            policy=policy,
        )
        self._states[deployment.id] = state
        self._deployments[deployment.id] = deployment

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

            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._watchdog.schedule_check(
                        deployment_id=deployment.id,
                        interval_seconds=interval,
                        callback=_watchdog_callback,
                    )
                )
            except RuntimeError:
                # No running event loop (e.g. sync creation); watchdog will be handled on async execution
                pass

        return state

    def record_activity(
        self, deployment_id: str, now: datetime | None = None
    ) -> None:
        """Records client activity or request handling on the specified deployment, resetting idle timer."""
        state = self._states.get(deployment_id)
        if state:
            state.last_activity_at = now or datetime.now(UTC)

    def is_idle(
        self, deployment_id: str, now: datetime | None = None
    ) -> bool:
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
            state.is_stopped = True
            state.stopped_at = now or datetime.now(UTC)

            try:
                await deployment.stop()
            except Exception as err:  # noqa: BLE001
                logger.error(
                    "Failed to stop deployment '%s' during autostop: %s",
                    deployment_id,
                    err,
                )
            finally:
                await self._watchdog.cancel_check(deployment_id)

            return True

        return False

    def get_state(self, deployment_id: str) -> LifecycleState | None:
        """Returns the current LifecycleState for the given deployment ID."""
        return self._states.get(deployment_id)

    async def unregister_deployment(self, deployment_id: str) -> None:
        """Unregisters deployment from lifecycle monitoring and cancels background watchdog tasks."""
        self._states.pop(deployment_id, None)
        self._deployments.pop(deployment_id, None)
        await self._watchdog.cancel_check(deployment_id)

    async def close(self) -> None:
        """Releases watchdog resources and cancels all scheduled checks."""
        await self._watchdog.close()
