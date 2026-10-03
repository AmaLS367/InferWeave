from abc import abstractmethod

from inferweave.domain.lifecycle import AutostopAction
from inferweave.ports.deployment_repository import DeploymentRepositoryPort
from inferweave.ports.provider import ComputeProviderPort


class ComputeProvider(ComputeProviderPort):
    """Abstract interface implemented by all infrastructure backends (SkyPilot, Modal, Docker)."""

    cleanup_failed_deployment = False

    def bind_repository(self, repository: DeploymentRepositoryPort) -> None:
        """Allows adapters requiring persisted resource identity to share SDK storage."""

    @abstractmethod
    async def stop(
        self,
        deployment_id: str,
        action: AutostopAction = AutostopAction.STOP,
    ) -> None:
        """Terminates or pauses the remote deployment."""
