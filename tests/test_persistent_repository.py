"""Unit tests for JsonDeploymentRepository persistent storage."""

import json
import os
import sqlite3
from pathlib import Path

import pytest

from inferweave.adapters.lifecycle.json_repository import JsonDeploymentRepository
from inferweave.domain.deployment_record import DeploymentRecord, ResourceRef
from inferweave.models.enums import DeploymentState


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
@pytest.mark.parametrize("owned", [True, False])
async def test_legacy_lightning_record_migrates_on_load(tmp_path, backend, owned):
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )

    legacy = {
        "id": "iw-old-lightning", "model": "fish-s2-pro", "provider": "lightning",
        "state": "healthy", "endpoint_url": "https://old.cloudspaces.litng.ai",
        "lightning": {
            "name": "old-deployment", "teamspace": "owner/original-space",
            "resource_id": "dep-old", "owned": owned,
        },
    }
    encoded = json.dumps(legacy)
    if backend == "json":
        path = tmp_path / "deployments.json"
        path.write_text(json.dumps({legacy["id"]: legacy}), encoding="utf-8")
        repo = JsonDeploymentRepository(path)
    else:
        path = tmp_path / "deployments.db"
        repo = SqliteDeploymentRepository(path)
        await repo.save(DeploymentRecord(id=legacy["id"], model=legacy["model"], provider="lightning"))
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE deployments SET data_json = ? WHERE id = ?", (encoded, legacy["id"]))

    record = await repo.get(legacy["id"])
    expected = ResourceRef(
        name="old-deployment", scope="owner/original-space", resource_id="dep-old", owned=owned,
    )
    assert record.resource == expected
    assert record.account == "ambient"
    assert (await repo.list_all())[0].resource == expected
    assert DeploymentRecord.model_validate(legacy).resource == expected
    assert json.dumps(legacy) == encoded  # validation must not mutate caller-owned input
    await repo.save(record)
    persisted = (await repo.get(record.id)).model_dump(mode="json")
    assert persisted["resource"] == expected.model_dump()
    assert "lightning" not in persisted


@pytest.mark.parametrize("resource", [None, {"name": "current", "scope": "new/space", "owned": False}])
def test_legacy_lightning_migration_preserves_current_resource(resource):
    record = DeploymentRecord.model_validate({
        "id": "old", "model": "fish-s2-pro", "provider": "lightning", "resource": resource,
        "lightning": {"name": "legacy", "teamspace": "old/space"},
    })
    if resource is None:
        assert record.resource == ResourceRef(name="legacy", scope="old/space")
    else:
        assert record.resource == ResourceRef.model_validate(resource)


def test_record_without_legacy_lightning_identity_still_loads():
    record = DeploymentRecord.model_validate({
        "id": "old", "model": "fish-s2-pro", "provider": "lightning", "lightning": None,
    })
    assert record.resource is None and record.account == "ambient"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_profile_snapshot_round_trip_and_sanitization(tmp_path: Path, backend: str):
    """Validates that profile_snapshot round-trips through JSON and SQLite with secrets sanitized."""
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )
    from inferweave.models.profile import HealthcheckConfig, ModelProfile

    repo = (
        JsonDeploymentRepository(tmp_path / "deployments.json")
        if backend == "json"
        else SqliteDeploymentRepository(tmp_path / "deployments.db")
    )

    clean_profile = ModelProfile.for_container(
        id="custom/token-classifier-model",
        image="ghcr.io/org/token-classifier:24.08-py3",
        args=[
            "tritonserver",
            "--model-repository=/models",
            "--tokenizer",
            "fast",
            "--max-new-tokens=512",
            "--eos-token-id",
            "2",
        ],
        port=8001,
        health_path="/v2/health/ready",
        setup_commands=["mkdir -p /models"],
        env={"TRITON_SERVER_CPU_ONLY": "0"},
        min_vram_gb=24.0,
        metadata={"revision": "r1", "tags": ["triton", "custom"]},
    )
    rec = DeploymentRecord(
        id="iw-roundtrip-1",
        model=clean_profile.id,
        provider="modal",
        state=DeploymentState.HEALTHY,
        endpoint_url="https://iw-roundtrip.modal.run",
        profile_snapshot=clean_profile,
    )
    await repo.save(rec)

    loaded = await repo.get("iw-roundtrip-1")
    assert loaded is not None
    assert isinstance(loaded.profile_snapshot, ModelProfile)
    assert loaded.profile_snapshot == clean_profile

    # Now test with secret-bearing profile fields
    secret_profile = ModelProfile.for_container(
        id="custom/secret-roundtrip",
        image="https://user:pass123@registry.example.com/org/image:latest",
        args=["--api-key=sk-secret-arg-987654", "--token", "hf_secret_token_abcdef"],
        port=8000,
        healthcheck=HealthcheckConfig(
            port=8000,
            path="/health?token=secret_query_token",
            headers={
                "Authorization": "Bearer sk-header-secret-111222",
                "X-Custom-Header": "normal-value",
            },
        ),
        setup_commands=[
            "curl -H 'Authorization: Bearer sk-setup-secret-333444' -H 'X-Auth-Token: custom_raw_secret_777' https://example.com"
        ],
        metadata={"credentials": {"password": "my-db-password-555"}},
    )
    secret_rec = DeploymentRecord(
        id="iw-roundtrip-secret",
        model=secret_profile.id,
        provider="modal",
        state=DeploymentState.HEALTHY,
        profile_snapshot=secret_profile,
    )
    await repo.save(secret_rec)

    raw_bytes = (
        (tmp_path / "deployments.json").read_bytes()
        if backend == "json"
        else (tmp_path / "deployments.db").read_bytes()
    )
    for leaked in (
        b"pass123",
        b"sk-secret-arg-987654",
        b"hf_secret_token_abcdef",
        b"secret_query_token",
        b"sk-header-secret-111222",
        b"sk-setup-secret-333444",
        b"custom_raw_secret_777",
        b"my-db-password-555",
    ):
        assert leaked not in raw_bytes

    loaded_secret = await repo.get("iw-roundtrip-secret")
    assert loaded_secret is not None
    assert isinstance(loaded_secret.profile_snapshot, ModelProfile)
    assert loaded_secret.profile_snapshot.healthcheck.headers["Authorization"] == "[REDACTED]"
    assert loaded_secret.profile_snapshot.healthcheck.headers["X-Custom-Header"] == "normal-value"
