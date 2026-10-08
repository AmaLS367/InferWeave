"""Port contract for persisting, retrieving, and updating deployment records."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager

from inferweave.domain.deployment_record import DeploymentRecord


class DeploymentRepositoryPort(ABC):
    """Abstract repository for persisting deployment metadata and lifecycle state."""

    @contextmanager
    def operation_lock(self, deployment_id: str) -> Iterator[bool]:
        """Nonblocking exclusion for provisioning/cleanup (custom durable stores must override)."""
        active = self.__dict__.setdefault("_active_operations", set())
        if deployment_id in active:
            yield False
            return
        active.add(deployment_id)
        try:
            yield True
        finally:
            active.remove(deployment_id)

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
