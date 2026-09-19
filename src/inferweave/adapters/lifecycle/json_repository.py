"""JSON file-based implementation of DeploymentRepositoryPort for cross-process persistence."""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.ports.deployment_repository import DeploymentRepositoryPort

logger = logging.getLogger(__name__)

DEFAULT_DEPLOYMENTS_FILE = Path.home() / ".inferweave" / "deployments.json"


class JsonDeploymentRepository(DeploymentRepositoryPort):
    """Persists deployment records to a local JSON file across distinct CLI invocations and processes."""

    def __init__(self, file_path: Path | str | None = None) -> None:
        if file_path is not None:
            self.file_path = Path(file_path)
        else:
            env_path = os.environ.get("INFERWEAVE_DEPLOYMENTS_PATH")
            self.file_path = Path(env_path) if env_path else DEFAULT_DEPLOYMENTS_FILE

        self._lock = asyncio.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensures that the directory enclosing the storage file exists."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            logger.warning("Could not create directory for deployments file: %s", err)

    def _read_records_sync(self) -> dict[str, DeploymentRecord]:
        """Synchronously reads and parses records from the JSON file."""
        if not self.file_path.exists():
            return {}

        try:
            content = self.file_path.read_text(encoding="utf-8").strip()
            if not content:
                return {}
            raw_data = json.loads(content)
            if not isinstance(raw_data, dict):
                logger.warning(
                    "Deployments file at '%s' has unexpected format. Expected dict, got %s.",
                    self.file_path,
                    type(raw_data),
                )
                return {}

            records: dict[str, DeploymentRecord] = {}
            for dep_id, item in raw_data.items():
                try:
                    records[dep_id] = DeploymentRecord.model_validate(item)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to validate deployment record '%s': %s", dep_id, exc
                    )
            return records
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt JSON in '%s': %s", self.file_path, exc)
            return {}
        except OSError as exc:
            logger.warning("Failed to read deployments file '%s': %s", self.file_path, exc)
            return {}

    def _write_records_sync(self, records: dict[str, DeploymentRecord]) -> None:
        """Atomically writes records to the JSON file via a temporary file."""
        self._ensure_dir()
        serialized = {dep_id: rec.model_dump(mode="json") for dep_id, rec in records.items()}
        data = json.dumps(serialized, indent=2, default=str)

        # Write to temporary file in same directory and atomically replace
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.file_path.parent,
                encoding="utf-8",
                delete=False,
                suffix=".tmp",
            ) as tf:
                tf.write(data)
                temp_file = tf.name

            os.replace(temp_file, self.file_path)
        except Exception as exc:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            logger.error("Failed to write deployment records to '%s': %s", self.file_path, exc)
            raise

    async def save(self, record: DeploymentRecord) -> None:
        """Persists or updates a deployment record."""
        async with self._lock:
            records = await asyncio.to_thread(self._read_records_sync)
            records[record.id] = record.model_copy(deep=True)
            await asyncio.to_thread(self._write_records_sync, records)

    async def get(self, deployment_id: str) -> DeploymentRecord | None:
        """Retrieves a deployment record by ID, or None if not found."""
        async with self._lock:
            records = await asyncio.to_thread(self._read_records_sync)
            record = records.get(deployment_id)
            return record.model_copy(deep=True) if record else None

    async def list_all(self) -> list[DeploymentRecord]:
        """Returns all persisted deployment records."""
        async with self._lock:
            records = await asyncio.to_thread(self._read_records_sync)
            return [rec.model_copy(deep=True) for rec in records.values()]

    async def delete(self, deployment_id: str) -> None:
        """Removes a deployment record from storage."""
        async with self._lock:
            records = await asyncio.to_thread(self._read_records_sync)
            if deployment_id in records:
                del records[deployment_id]
                await asyncio.to_thread(self._write_records_sync, records)

    def clear(self) -> None:
        """Clears all records (useful for testing)."""
        self._write_records_sync({})
