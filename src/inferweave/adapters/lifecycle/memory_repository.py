"""In-memory implementation of DeploymentRepositoryPort."""

import asyncio

from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.ports.deployment_repository import DeploymentRepositoryPort


class InMemoryDeploymentRepository(DeploymentRepositoryPort):
    """Thread-safe, in-memory repository for storing and tracking active deployment records."""

    def __init__(self, initial_records: list[DeploymentRecord] | None = None) -> None:
        self._records: dict[str, DeploymentRecord] = {}
        self._lock = asyncio.Lock()
        if initial_records:
            for record in initial_records:
                self._records[record.id] = record.model_copy(deep=True)

    async def save(self, record: DeploymentRecord) -> None:
        """Saves or updates a deployment record in memory."""
        async with self._lock:
            self._records[record.id] = record.model_copy(deep=True)

    async def get(self, deployment_id: str) -> DeploymentRecord | None:
        """Retrieves a clone of the deployment record by ID, or None if not found."""
        async with self._lock:
            record = self._records.get(deployment_id)
            return record.model_copy(deep=True) if record else None

    async def list_all(self) -> list[DeploymentRecord]:
        """Returns all tracked deployment records."""
        async with self._lock:
            return [rec.model_copy(deep=True) for rec in self._records.values()]

    async def delete(self, deployment_id: str) -> None:
        """Removes a deployment record from memory."""
        async with self._lock:
            self._records.pop(deployment_id, None)

    def clear(self) -> None:
        """Clears all stored records (useful for testing)."""
        self._records.clear()
