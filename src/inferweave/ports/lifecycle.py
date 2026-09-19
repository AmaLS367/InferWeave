"""Port contract for scheduling and running autostop watchdog monitoring tasks."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any


class AutostopWatchdogPort(ABC):
    """Abstract port for scheduling periodic idle checks against active deployments."""

    @abstractmethod
    async def schedule_check(
        self,
        deployment_id: str,
        interval_seconds: float,
        callback: Callable[[], Coroutine[Any, Any, None]],
    ) -> None:
        """Schedules a recurring asynchronous idle verification task for a deployment."""

    @abstractmethod
    async def cancel_check(self, deployment_id: str) -> None:
        """Cancels any pending or recurring watchdog checks for the specified deployment."""

    @abstractmethod
    async def close(self) -> None:
        """Terminates all running watchdog tasks and releases resources."""
