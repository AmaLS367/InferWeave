from abc import abstractmethod

from inferweave.domain.lifecycle import AutostopAction
from inferweave.ports.provider import ComputeProviderPort


class ComputeProvider(ComputeProviderPort):
    """Abstract interface implemented by all infrastructure backends (SkyPilot, Modal, Docker)."""

    @abstractmethod
    async def stop(
        self,
        deployment_id: str,
        action: AutostopAction = AutostopAction.STOP,
    ) -> None:
        """Terminates or pauses the remote deployment."""
