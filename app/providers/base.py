"""Compute provider abstract base class."""

from abc import ABC, abstractmethod

from app.models.deployment import Deployment, DeploymentRequest, DeploymentStatus
from app.models.enums import ProviderType
from app.models.profile import ModelProfile
from app.runtimes.base import RuntimeSpec


class ComputeProvider(ABC):
    """Abstract interface implemented by all infrastructure backends (SkyPilot, Modal, Docker)."""

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
    async def stop(self, deployment_id: str) -> None:
        """Terminates or pauses the remote deployment."""

    @abstractmethod
    async def get_status(self, deployment_id: str) -> DeploymentStatus:
        """Fetches the latest status and endpoint health for a given deployment ID."""
