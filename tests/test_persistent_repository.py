"""Unit tests for JsonDeploymentRepository persistent storage."""

import os
from pathlib import Path

import pytest

from inferweave.adapters.lifecycle.json_repository import JsonDeploymentRepository
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.models.enums import DeploymentState


@pytest.fixture
def temp_repo_file(tmp_path: Path) -> Path:
    return tmp_path / "deployments.json"


@pytest.fixture
def sample_record() -> DeploymentRecord:
    return DeploymentRecord(
        id="iw-test-123",
        model="fish-s2-pro",
        provider="modal",
        state=DeploymentState.HEALTHY,
        endpoint_url="https://test.modal.run",
    )


@pytest.mark.asyncio
async def test_json_repo_crud(temp_repo_file: Path, sample_record: DeploymentRecord):
    repo = JsonDeploymentRepository(file_path=temp_repo_file)

    # 1. Initially empty
    records = await repo.list_all()
    assert records == []
    assert await repo.get("non-existent") is None

    # 2. Save record
    await repo.save(sample_record)
    assert temp_repo_file.exists()

    # 3. Retrieve record
    retrieved = await repo.get(sample_record.id)
    assert retrieved is not None
    assert retrieved.id == sample_record.id
    assert retrieved.model == "fish-s2-pro"
    assert retrieved.provider == "modal"
    assert retrieved.state == DeploymentState.HEALTHY

    # 4. List records
    all_records = await repo.list_all()
    assert len(all_records) == 1
    assert all_records[0].id == sample_record.id

    # 5. Update record
    sample_record.mark_stopped()
    await repo.save(sample_record)
    updated = await repo.get(sample_record.id)
    assert updated is not None
    assert updated.state == DeploymentState.STOPPED

    # 6. Delete record
    await repo.delete(sample_record.id)
    assert await repo.get(sample_record.id) is None
    assert await repo.list_all() == []


@pytest.mark.asyncio
async def test_json_repo_env_var_path(tmp_path: Path, sample_record: DeploymentRecord):
    env_file = tmp_path / "env_deployments.json"
    os.environ["INFERWEAVE_DEPLOYMENTS_PATH"] = str(env_file)
    try:
        repo = JsonDeploymentRepository()
        assert repo.file_path == env_file
        await repo.save(sample_record)
        assert env_file.exists()
        rec = await repo.get(sample_record.id)
        assert rec is not None
        assert rec.id == sample_record.id
    finally:
        del os.environ["INFERWEAVE_DEPLOYMENTS_PATH"]


@pytest.mark.asyncio
async def test_json_repo_corrupt_file_recovery(temp_repo_file: Path):
    temp_repo_file.write_text("invalid json content {{{", encoding="utf-8")
    repo = JsonDeploymentRepository(file_path=temp_repo_file)
    assert await repo.list_all() == []
    assert await repo.get("any") is None


@pytest.mark.asyncio
async def test_json_repo_clear(temp_repo_file: Path, sample_record: DeploymentRecord):
    repo = JsonDeploymentRepository(file_path=temp_repo_file)
    await repo.save(sample_record)
    assert len(await repo.list_all()) == 1
    repo.clear()
    assert len(await repo.list_all()) == 0


@pytest.fixture
def temp_sqlite_file(tmp_path: Path) -> Path:
    return tmp_path / "deployments.db"


@pytest.mark.asyncio
async def test_sqlite_repo_crud(temp_sqlite_file: Path, sample_record: DeploymentRecord):
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )

    repo = SqliteDeploymentRepository(db_path=temp_sqlite_file)

    # 1. Initially empty
    records = await repo.list_all()
    assert records == []
    assert await repo.get("non-existent") is None

    # 2. Save record
    await repo.save(sample_record)
    assert temp_sqlite_file.exists()

    # 3. Retrieve record
    retrieved = await repo.get(sample_record.id)
    assert retrieved is not None
    assert retrieved.id == sample_record.id
    assert retrieved.model == "fish-s2-pro"
    assert retrieved.provider == "modal"
    assert retrieved.state == DeploymentState.HEALTHY

    # 4. List records
    all_records = await repo.list_all()
    assert len(all_records) == 1
    assert all_records[0].id == sample_record.id

    # 5. Update record
    sample_record.mark_stopped()
    await repo.save(sample_record)
    updated = await repo.get(sample_record.id)
    assert updated is not None
    assert updated.state == DeploymentState.STOPPED

    # 6. Delete record
    await repo.delete(sample_record.id)
    assert await repo.get(sample_record.id) is None
    assert await repo.list_all() == []


@pytest.mark.asyncio
async def test_sqlite_repo_env_var_path(tmp_path: Path, sample_record: DeploymentRecord):
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )

    env_file = tmp_path / "env_deployments.db"
    os.environ["INFERWEAVE_DEPLOYMENTS_PATH"] = str(env_file)
    try:
        repo = SqliteDeploymentRepository()
        assert repo.db_path == env_file
        await repo.save(sample_record)
        assert env_file.exists()
        rec = await repo.get(sample_record.id)
        assert rec is not None
        assert rec.id == sample_record.id
    finally:
        del os.environ["INFERWEAVE_DEPLOYMENTS_PATH"]


@pytest.mark.asyncio
async def test_sqlite_repo_clear(temp_sqlite_file: Path, sample_record: DeploymentRecord):
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )

    repo = SqliteDeploymentRepository(db_path=temp_sqlite_file)
    await repo.save(sample_record)
    assert len(await repo.list_all()) == 1
    repo.clear()
    assert len(await repo.list_all()) == 0


@pytest.mark.asyncio
async def test_sqlite_repo_concurrent_writes(temp_sqlite_file: Path):
    import asyncio

    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )

    repo = SqliteDeploymentRepository(db_path=temp_sqlite_file)

    async def write_task(idx: int):
        rec = DeploymentRecord(
            id=f"iw-concurrent-{idx}",
            model="fish-s2-pro",
            provider="modal",
            state=DeploymentState.HEALTHY,
        )
        await repo.save(rec)

    # Launch 20 concurrent saves
    await asyncio.gather(*(write_task(i) for i in range(20)))

    all_records = await repo.list_all()
    assert len(all_records) == 20
    ids = {r.id for r in all_records}
    assert len(ids) == 20


@pytest.mark.asyncio
async def test_sqlite_repo_sanitizes_secrets(temp_sqlite_file: Path):
    """Validates that credentials and secrets are redacted before persistence to disk."""
    import sqlite3

    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )
    from inferweave.domain.options import (
        DeploymentOptions,
        ProviderOptions,
        RuntimeOptions,
    )

    repo = SqliteDeploymentRepository(db_path=temp_sqlite_file)
    rec = DeploymentRecord(
        id="iw-secrets-test-1",
        model="fish-s2-pro",
        provider="modal",
        options=DeploymentOptions(
            provider=ProviderOptions(
                extra_provider_args={
                    "secrets": {"HF_TOKEN": "hf_secret_12345", "MODAL_SECRET": "secret_xyz"},
                    "api_key": "raw_secret_key",
                }
            ),
            runtime=RuntimeOptions(
                extra_env={"HF_TOKEN": "hf_runtime_env_secret", "NORMAL_VAR": "hello"}
            ),
        ),
    )

    await repo.save(rec)

    # Read the raw SQLite database directly to verify on-disk content
    conn = sqlite3.connect(str(temp_sqlite_file))
    cursor = conn.cursor()
    cursor.execute("SELECT data_json FROM deployments WHERE id = 'iw-secrets-test-1';")
    raw_data_json = cursor.fetchone()[0]
    conn.close()

    # Sensitive values must NOT appear in raw stored JSON
    assert "hf_secret_12345" not in raw_data_json
    assert "secret_xyz" not in raw_data_json
    assert "raw_secret_key" not in raw_data_json
    assert "hf_runtime_env_secret" not in raw_data_json

    # Values must be redacted
    assert "[REDACTED]" in raw_data_json
    assert "NORMAL_VAR" in raw_data_json
    assert "hello" in raw_data_json

