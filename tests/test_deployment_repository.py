"""Unit tests for DeploymentRepositoryPort and InMemoryDeploymentRepository."""

import pytest

from inferweave.adapters.lifecycle.memory_repository import InMemoryDeploymentRepository
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.options import DeploymentOptions
from inferweave.models.enums import DeploymentState


@pytest.mark.asyncio
async def test_in_memory_repository_crud():
    repo = InMemoryDeploymentRepository()

    record = DeploymentRecord(
        id="iw-test-123",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        provider="runpod",
        state=DeploymentState.PROVISIONING,
        endpoint_url="http://1.2.3.4:8000",
        options=DeploymentOptions(),
        is_dry_run=False,
    )

    # 1. Save
    await repo.save(record)

    # 2. Get
    fetched = await repo.get("iw-test-123")
    assert fetched is not None
    assert fetched.id == "iw-test-123"
    assert fetched.model == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert fetched.state == DeploymentState.PROVISIONING
    assert fetched.is_active() is True

    # 3. Update status to HEALTHY
    fetched.mark_healthy()
    await repo.save(fetched)

    updated = await repo.get("iw-test-123")
    assert updated is not None
    assert updated.state == DeploymentState.HEALTHY
    assert updated.ready_at is not None

    # 4. List all
    all_records = await repo.list_all()
    assert len(all_records) == 1
    assert all_records[0].id == "iw-test-123"

    # 5. Mark stopped
    updated.mark_stopped()
    await repo.save(updated)
    stopped = await repo.get("iw-test-123")
    assert stopped is not None
    assert stopped.state == DeploymentState.STOPPED
    assert stopped.stopped_at is not None
    assert stopped.is_active() is False

    # 6. Delete
    await repo.delete("iw-test-123")
    assert await repo.get("iw-test-123") is None
    assert len(await repo.list_all()) == 0


@pytest.mark.asyncio
async def test_deployment_record_failure_and_degradation():
    record = DeploymentRecord(
        id="iw-test-456",
        model="fish-s2-pro",
        provider="modal",
    )
    assert record.is_active() is True

    record.mark_degraded(error_message="High latency")
    assert record.state == DeploymentState.DEGRADED
    assert record.error_message == "High latency"
    assert record.is_active() is True

    record.mark_failed(error_message="OOM crash")
    assert record.state == DeploymentState.FAILED
    assert record.error_message == "OOM crash"
    assert record.stopped_at is not None
    assert record.is_active() is False
