"""Port contract for cloud and container compute infrastructure backends."""

from abc import ABC, abstractmethod

from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from inferweave.models.enums import ProviderType
from inferweave.models.profile import ModelProfile
from inferweave.runtimes.base import RuntimeSpec


class ComputeProviderPort(ABC):
    """Abstract port interface for infrastructure backends (SkyPilot, Modal, Docker)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for this provider instance (e.g. 'runpod', 'modal', 'aws')."""

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Category of this compute backend."""

    @abstractmethod
    async def deploy(
        self,
        request: DeploymentRequest,
        profile: ModelProfile,
        runtime: RuntimeSpec,
    ) -> Deployment:
        """Provisions hardware, launches the containerized runtime, and returns an active Deployment."""

    @abstractmethod
    async def stop(
        self,
        deployment_id: str,
        action: AutostopAction = AutostopAction.STOP,
    ) -> None:
        """Terminates or pauses the remote deployment."""

    @abstractmethod
    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Fetches the latest status and endpoint health for a given deployment ID."""
