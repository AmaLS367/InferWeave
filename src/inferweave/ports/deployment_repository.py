"""Port contract for persisting, retrieving, and updating deployment records."""

from abc import ABC, abstractmethod

from inferweave.domain.deployment_record import DeploymentRecord


class DeploymentRepositoryPort(ABC):
    """Abstract repository for persisting deployment metadata and lifecycle state."""

    @abstractmethod
    async def save(self, record: DeploymentRecord) -> None:
        """Stores or updates a deployment record."""

    @abstractmethod
    async def get(self, deployment_id: str) -> DeploymentRecord | None:
        """Retrieves a deployment record by its identifier, or None if not found."""

    @abstractmethod
    async def list_all(self) -> list[DeploymentRecord]:
        """Returns all tracked deployment records."""

    @abstractmethod
    async def delete(self, deployment_id: str) -> None:
        """Removes a deployment record from tracking."""
