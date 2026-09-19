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
