"""Concrete adapters implementing AutostopWatchdogPort."""

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from inferweave.ports.lifecycle import AutostopWatchdogPort

logger = logging.getLogger(__name__)


class AsyncioWatchdogAdapter(AutostopWatchdogPort):
    """Production watchdog adapter using background asyncio tasks for idle monitoring."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def schedule_check(
        self,
        deployment_id: str,
        interval_seconds: float,
        callback: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Schedules a recurring background task verifying deployment idle state."""
        await self.cancel_check(deployment_id)

        async def _loop() -> None:
            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await callback()
                except asyncio.CancelledError:
                    break
                except Exception as err:  # noqa: BLE001
                    logger.warning(
                        "Error during watchdog execution for deployment '%s': %s",
                        deployment_id,
                        err,
                    )

        task = asyncio.create_task(_loop())
        self._tasks[deployment_id] = task

    async def cancel_check(self, deployment_id: str) -> None:
        """Cancels and cleans up the watchdog task for the specified deployment."""
        task = self._tasks.pop(deployment_id, None)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def close(self) -> None:
        """Cancels all active watchdog tasks."""
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class MockWatchdogAdapter(AutostopWatchdogPort):
    """Deterministic in-memory watchdog adapter for unit testing without asyncio delays."""

    def __init__(self) -> None:
        self.scheduled_checks: dict[
            str, tuple[float, Callable[[], Coroutine[Any, Any, None]]]
        ] = {}
        self.cancelled_checks: list[str] = []

    async def schedule_check(
        self,
        deployment_id: str,
        interval_seconds: float,
        callback: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        self.scheduled_checks[deployment_id] = (interval_seconds, callback)

    async def cancel_check(self, deployment_id: str) -> None:
        self.scheduled_checks.pop(deployment_id, None)
        self.cancelled_checks.append(deployment_id)

    async def trigger_check(self, deployment_id: str) -> None:
        """Manually and synchronously triggers the registered callback for a deployment."""
        entry = self.scheduled_checks.get(deployment_id)
        if entry:
            _, callback = entry
            await callback()

    async def close(self) -> None:
        self.scheduled_checks.clear()
