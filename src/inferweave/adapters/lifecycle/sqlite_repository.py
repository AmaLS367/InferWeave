"""SQLite-based implementation of DeploymentRepositoryPort for transactional, multi-process persistence."""

import asyncio
import logging
import os
import sqlite3
from pathlib import Path

from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.ports.deployment_repository import DeploymentRepositoryPort

logger = logging.getLogger(__name__)

DEFAULT_DEPLOYMENTS_DB = Path.home() / ".inferweave" / "deployments.db"


class SqliteDeploymentRepository(DeploymentRepositoryPort):
    """Persists deployment records in an SQLite database with ACID transactions and WAL mode."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            env_path = os.environ.get("INFERWEAVE_DEPLOYMENTS_PATH")
            self.db_path = Path(env_path) if env_path else DEFAULT_DEPLOYMENTS_DB

        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Initializes database directory and tables."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS deployments (
                        id TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        state TEXT NOT NULL,
                        endpoint_url TEXT,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        ready_at TEXT,
                        stopped_at TEXT,
                        is_dry_run INTEGER NOT NULL DEFAULT 0,
                        data_json TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
        except OSError as err:
            logger.warning("Could not initialize database directory at '%s': %s", self.db_path, err)

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes an SQLite connection with reasonable busy timeout for concurrency."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _save_sync(self, record: DeploymentRecord) -> None:
        sanitized = record.to_sanitized_record()
        data_json = sanitized.model_dump_json()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO deployments (
                    id, model, provider, state, endpoint_url, error_message,
                    created_at, ready_at, stopped_at, is_dry_run, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    model=excluded.model,
                    provider=excluded.provider,
                    state=excluded.state,
                    endpoint_url=excluded.endpoint_url,
                    error_message=excluded.error_message,
                    created_at=excluded.created_at,
                    ready_at=excluded.ready_at,
                    stopped_at=excluded.stopped_at,
                    is_dry_run=excluded.is_dry_run,
                    data_json=excluded.data_json;
                """,
                (
                    record.id,
                    record.model,
                    record.provider,
                    record.state.value if hasattr(record.state, "value") else str(record.state),
                    record.endpoint_url,
                    record.error_message,
                    record.created_at.isoformat(),
                    record.ready_at.isoformat() if record.ready_at else None,
                    record.stopped_at.isoformat() if record.stopped_at else None,
                    1 if record.is_dry_run else 0,
                    data_json,
                ),
            )
            conn.commit()

    def _get_sync(self, deployment_id: str) -> DeploymentRecord | None:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT data_json FROM deployments WHERE id = ?;", (deployment_id,))
            row = cursor.fetchone()
            if not row:
                return None
            try:
                return DeploymentRecord.model_validate_json(row["data_json"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse deployment record '%s': %s", deployment_id, exc)
                return None

    def _list_all_sync(self) -> list[DeploymentRecord]:
        records: list[DeploymentRecord] = []
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT data_json FROM deployments ORDER BY created_at DESC;")
            for row in cursor.fetchall():
                try:
                    records.append(DeploymentRecord.model_validate_json(row["data_json"]))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to parse record in list_all: %s", exc)
        return records

    def _delete_sync(self, deployment_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM deployments WHERE id = ?;", (deployment_id,))
            conn.commit()

    def _clear_sync(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM deployments;")
            conn.commit()

    async def save(self, record: DeploymentRecord) -> None:
        """Stores or updates a deployment record asynchronously."""
        await asyncio.to_thread(self._save_sync, record.model_copy(deep=True))

    async def get(self, deployment_id: str) -> DeploymentRecord | None:
        """Retrieves a deployment record by ID, or None if not found."""
        return await asyncio.to_thread(self._get_sync, deployment_id)

    async def list_all(self) -> list[DeploymentRecord]:
        """Returns all persisted deployment records."""
        return await asyncio.to_thread(self._list_all_sync)

    async def delete(self, deployment_id: str) -> None:
        """Removes a deployment record from SQLite storage."""
        await asyncio.to_thread(self._delete_sync, deployment_id)

    def clear(self) -> None:
        """Clears all records synchronously (useful for test fixtures)."""
        self._clear_sync()
