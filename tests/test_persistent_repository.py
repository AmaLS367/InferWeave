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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_extra_provider_args_secrets_redacts_opaque_strings_and_structures(
    tmp_path: Path, backend: str
):
    """Verifies that all values under extra_provider_args['secrets'] are redacted, including opaque strings."""
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )
    from inferweave.domain.options import DeploymentOptions, ProviderOptions

    repo = (
        JsonDeploymentRepository(tmp_path / "deployments.json")
        if backend == "json"
        else SqliteDeploymentRepository(tmp_path / "deployments.db")
    )

    opaque_secret_1 = "OPAQUE_RAW_SECRET_WITHOUT_EQUALS_XYZ123"
    opaque_secret_2 = "sk-opaque-secret-val-456"
    opaque_secret_3 = "JUST_AN_OPAQUE_TOKEN_999"
    opaque_in_dict = "opaque_in_dict_888"
    nested_secret_str = "nested_secret_str_777"
    sensitive_extra_arg = "sensitive_extra_arg_111"

    rec = DeploymentRecord(
        id="iw-secrets-opaque-test",
        model="custom/test-model",
        provider="modal",
        state=DeploymentState.HEALTHY,
        options=DeploymentOptions(
            provider=ProviderOptions(
                extra_provider_args={
                    "secrets": [
                        opaque_secret_1,
                        f"API_KEY={opaque_secret_2}",
                        opaque_secret_3,
                        {"SECRET_KEY": opaque_in_dict},
                        [nested_secret_str],
                    ],
                    "other_secret": sensitive_extra_arg,
                }
            )
        ),
    )

    await repo.save(rec)

    raw_bytes = (
        (tmp_path / "deployments.json").read_bytes()
        if backend == "json"
        else (tmp_path / "deployments.db").read_bytes()
    )

    # Raw bytes must never contain any of these secret strings
    for secret in (
        opaque_secret_1.encode("utf-8"),
        opaque_secret_2.encode("utf-8"),
        opaque_secret_3.encode("utf-8"),
        opaque_in_dict.encode("utf-8"),
        nested_secret_str.encode("utf-8"),
        sensitive_extra_arg.encode("utf-8"),
    ):
        assert secret not in raw_bytes

    # repr and str of record must not leak secrets
    assert opaque_secret_1 not in repr(rec)
    assert opaque_secret_2 not in repr(rec)
    assert opaque_secret_3 not in repr(rec)
    assert opaque_in_dict not in repr(rec)
    assert nested_secret_str not in repr(rec)
    assert sensitive_extra_arg not in repr(rec)

    loaded = await repo.get("iw-secrets-opaque-test")
    assert loaded is not None
    loaded_secrets = loaded.options.provider.extra_provider_args["secrets"]
    assert loaded_secrets[0] == "[REDACTED]"
    assert loaded_secrets[1] == "API_KEY=[REDACTED]"
    assert loaded_secrets[2] == "[REDACTED]"
    assert loaded_secrets[3] == {"SECRET_KEY": "[REDACTED]"}
    assert loaded_secrets[4] == ["[REDACTED]"]
    assert loaded.options.provider.extra_provider_args["other_secret"] == "[REDACTED]"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_nested_metadata_generic_keys_sanitized_without_destroying_ordinary_metadata(
    tmp_path: Path, backend: str
):
    """Verifies generic secret-bearing keys are redacted while preserving ordinary metadata."""
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )
    from inferweave.domain.options import DeploymentOptions, ProviderOptions
    from inferweave.models.profile import ModelProfile

    repo = (
        JsonDeploymentRepository(tmp_path / "deployments.json")
        if backend == "json"
        else SqliteDeploymentRepository(tmp_path / "deployments.db")
    )

    secret_key_val = "generic_secret_key_val_123"
    secret_k1 = "generic_secret_k1"
    secret_k2 = "generic_secret_k2"
    secret_signing = "sensitive_prefix_key_456"
    secret_tag_val = "sk-should-be-redacted-val-333"

    profile = ModelProfile.for_container(
        id="custom/metadata-generic-key-model",
        image="ghcr.io/org/test:latest",
        metadata={
            "key": secret_key_val,
            "keys": [secret_k1, secret_k2],
            "signing_key": secret_signing,
            "sort_key": "ascending_order_term",
            "cache_key": "user_session_cache_id_789",
            "primary_key": "row_id_primary",
            "partition_key": "shard_001",
            "key_name": "tag_name_cluster",
            "tags": [
                {"key": "environment", "value": "production_cluster_use_case"},
                {"key": "service", "value": "inference_gateway"},
                {"key": "api_key", "value": secret_tag_val},
            ],
        },
    )

    rec = DeploymentRecord(
        id="iw-metadata-test",
        model=profile.id,
        provider="modal",
        state=DeploymentState.HEALTHY,
        profile_snapshot=profile,
        options=DeploymentOptions(
            provider=ProviderOptions(
                extra_provider_args={
                    "metadata": {
                        "key": secret_key_val,
                        "sort_key": "descending",
                    }
                }
            )
        ),
    )

    await repo.save(rec)

    raw_bytes = (
        (tmp_path / "deployments.json").read_bytes()
        if backend == "json"
        else (tmp_path / "deployments.db").read_bytes()
    )

    # Sensitive values must never appear in raw bytes
    for sec in (
        secret_key_val.encode("utf-8"),
        secret_k1.encode("utf-8"),
        secret_k2.encode("utf-8"),
        secret_signing.encode("utf-8"),
        secret_tag_val.encode("utf-8"),
    ):
        assert sec not in raw_bytes

    # Benign values MUST be preserved in raw bytes
    for benign in (
        b"ascending_order_term",
        b"user_session_cache_id_789",
        b"row_id_primary",
        b"shard_001",
        b"tag_name_cluster",
        b"production_cluster_use_case",
        b"inference_gateway",
        b"descending",
    ):
        assert benign in raw_bytes

    loaded = await repo.get("iw-metadata-test")
    assert loaded is not None
    assert isinstance(loaded.profile_snapshot, ModelProfile)
    meta = loaded.profile_snapshot.metadata
    assert meta["key"] == "[REDACTED]"
    assert meta["keys"] == ["[REDACTED]", "[REDACTED]"]
    assert meta["signing_key"] == "[REDACTED]"
    assert meta["sort_key"] == "ascending_order_term"
    assert meta["cache_key"] == "user_session_cache_id_789"
    assert meta["primary_key"] == "row_id_primary"
    assert meta["partition_key"] == "shard_001"
    assert meta["key_name"] == "tag_name_cluster"
    assert meta["tags"][0] == {"key": "environment", "value": "production_cluster_use_case"}
    assert meta["tags"][1] == {"key": "service", "value": "inference_gateway"}
    assert meta["tags"][2] == {"key": "api_key", "value": "[REDACTED]"}

    # Provider extra args metadata
    prov_meta = loaded.options.provider.extra_provider_args["metadata"]
    assert prov_meta["key"] == "[REDACTED]"
    assert prov_meta["sort_key"] == "descending"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_all_model_profile_fields_sanitized_in_storage_and_repr(
    tmp_path: Path, backend: str
):
    """Verifies that all persisted ModelProfile fields and DeploymentRecord fields sanitize secrets."""
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )
    from inferweave.domain.options import DeploymentOptions, RuntimeOptions
    from inferweave.models.enums import WorkloadType
    from inferweave.models.profile import (
        ContainerSpec,
        HardwareRequirements,
        HealthcheckConfig,
        ModelProfile,
    )

    repo = (
        JsonDeploymentRepository(tmp_path / "deployments.json")
        if backend == "json"
        else SqliteDeploymentRepository(tmp_path / "deployments.db")
    )

    secrets = {
        "pass_source": "pass-source-456",
        "name_sec": "sk-name-secret-123",
        "art_sec": "sk-artifact-secret-789",
        "gpu_sec": "sk-gpu-secret-001",
        "cuda_sec": "sk-cuda-secret-002",
        "pass_img": "pass-img-333",
        "args_sec": "sk-args-secret-444",
        "env_sec": "sk-env-secret-555",
        "setup_sec": "sk-setup-secret-666",
        "def_env_sec": "sk-def-env-secret-777",
        "hc_path_sec": "sk-hc-path-secret-888",
        "hc_hdr_sec": "sk-hc-hdr-secret-999",
        "pass_ep": "pass-endpoint-000",
        "err_sec": "sk-err-msg-secret-111",
        "runtime_cli_sec": "sk-runtime-cli-secret-222",
        "runtime_eng_sec": "sk-engine-args-secret-333",
    }

    profile = ModelProfile(
        id="custom/clean-model-id",
        name=f"Model {secrets['name_sec']}",
        workload_type=WorkloadType.CUSTOM,
        default_runtime="container",
        source=f"https://user:{secrets['pass_source']}@huggingface.co/org/model",
        artifact_id=f"art-{secrets['art_sec']}",
        hardware=HardwareRequirements(
            min_vram_gb=16.0,
            recommended_gpus=[f"GPU {secrets['gpu_sec']}"],
            min_cuda_version=f"12.4 {secrets['cuda_sec']}",
        ),
        container=ContainerSpec(
            image=f"https://user:{secrets['pass_img']}@reg.io/img:latest",
            args=[f"--key={secrets['args_sec']}"],
            env={"APP_SETTING": f"prefix-{secrets['env_sec']}"},
            setup_commands=[f"echo {secrets['setup_sec']}"],
        ),
        default_env={"MODEL_CONFIG": f"val-{secrets['def_env_sec']}", "BENIGN_ENV": "benign_val"},
        healthcheck=HealthcheckConfig(
            port=8000,
            path=f"/health?token={secrets['hc_path_sec']}",
            headers={"X-Api-Key": secrets["hc_hdr_sec"]},
        ),
    )

    rec = DeploymentRecord(
        id="iw-audit-all-fields",
        model=profile.id,
        provider="modal",
        state=DeploymentState.HEALTHY,
        endpoint_url=f"https://user:{secrets['pass_ep']}@endpoint.modal.run",
        error_message=f"Failed with {secrets['err_sec']}",
        profile_snapshot=profile,
        options=DeploymentOptions(
            runtime=RuntimeOptions(
                extra_cli_args=[f"--secret-token={secrets['runtime_cli_sec']}"],
                engine_args={"api_token": secrets["runtime_eng_sec"]},
            )
        ),
    )

    # __repr__ and __str__ on ModelProfile and DeploymentRecord must not leak
    profile_repr = repr(profile)
    record_repr = repr(rec)
    for s_name, s_val in secrets.items():
        assert s_val not in profile_repr, f"Secret {s_name} leaked in profile repr"
        assert s_val not in record_repr, f"Secret {s_name} leaked in record repr"

    await repo.save(rec)

    raw_bytes = (
        (tmp_path / "deployments.json").read_bytes()
        if backend == "json"
        else (tmp_path / "deployments.db").read_bytes()
    )

    for s_name, s_val in secrets.items():
        assert s_val.encode("utf-8") not in raw_bytes, f"Secret {s_name} leaked in raw bytes"

    # SQLite relational columns audit
    if backend == "sqlite":
        with sqlite3.connect(tmp_path / "deployments.db") as conn:
            row = conn.execute(
                "SELECT endpoint_url, error_message, model FROM deployments WHERE id = 'iw-audit-all-fields'"
            ).fetchone()
            assert secrets["pass_ep"] not in (row[0] or "")
            assert secrets["err_sec"] not in (row[1] or "")


def test_model_profile_rejects_secret_in_required_identity_id():
    """Verifies that ModelProfile safely rejects secret tokens in required identity field 'id'."""
    from inferweave.models.profile import ModelProfile

    # Reject known secret prefixes / patterns
    with pytest.raises(ValueError, match="contains sensitive credentials or patterns"):
        ModelProfile.for_container(
            id="sk-secret-token-1234567890",
            image="alpine:latest",
        )

    with pytest.raises(ValueError, match="contains sensitive credentials or patterns"):
        ModelProfile.for_container(
            id="hf_abcxyz1234567890",
            image="alpine:latest",
        )

    with pytest.raises(ValueError, match="contains sensitive credentials or patterns"):
        ModelProfile.for_container(
            id="ghp_abc1234567890",
            image="alpine:latest",
        )

    with pytest.raises(ValueError, match="cannot contain redaction markers"):
        ModelProfile.for_container(
            id="custom/[REDACTED]",
            image="alpine:latest",
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        ModelProfile.for_container(
            id="   ",
            image="alpine:latest",
        )

    # Valid non-secret IDs with terms like token or secret must NOT be rejected
    p1 = ModelProfile.for_container(
        id="custom/token-classifier-model", image="alpine:latest"
    )
    assert p1.id == "custom/token-classifier-model"

    p2 = ModelProfile.for_container(
        id="custom/secret-sharing-scheme", image="alpine:latest"
    )
    assert p2.id == "custom/secret-sharing-scheme"

    # If explicitly known secret matches ID, to_sanitized must refuse to sanitize with ValueError
    with pytest.raises(ValueError, match="cannot be safely sanitized without breaking required identity"):
        p1.to_sanitized(secrets=["token-classifier"])


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["json", "sqlite"])
async def test_valid_snapshot_recovery_legacy_records_and_precedence(
    tmp_path: Path, backend: str
):
    """Verifies valid snapshot recovery, legacy records without snapshots, and registered profile precedence."""
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )
    from inferweave.core.exceptions import ModelNotFoundError
    from inferweave.domain.deployment_record import recover_profile_from_snapshot
    from inferweave.models.profile import ModelProfile
    from inferweave.registry.base import ModelRegistry

    repo = (
        JsonDeploymentRepository(tmp_path / "deployments.json")
        if backend == "json"
        else SqliteDeploymentRepository(tmp_path / "deployments.db")
    )

    clean_profile = ModelProfile.for_container(
        id="custom/valid-recovered-model",
        image="ghcr.io/org/valid:latest",
        env={"MODEL_ENV": "prod"},
    )

    # 1. Valid snapshot roundtrips and recovers
    valid_rec = DeploymentRecord(
        id="iw-valid-rec",
        model=clean_profile.id,
        provider="modal",
        state=DeploymentState.HEALTHY,
        profile_snapshot=clean_profile,
    )
    await repo.save(valid_rec)

    loaded = await repo.get("iw-valid-rec")
    assert loaded is not None
    recovered = recover_profile_from_snapshot(loaded)
    assert recovered == clean_profile

    # 2. Legacy record without snapshot raises ModelNotFoundError on snapshot recovery
    legacy_rec = DeploymentRecord(
        id="iw-legacy-rec",
        model="registered/llama-3-8b",
        provider="modal",
        state=DeploymentState.HEALTHY,
        profile_snapshot=None,
    )
    await repo.save(legacy_rec)

    loaded_legacy = await repo.get("iw-legacy-rec")
    assert loaded_legacy is not None
    assert loaded_legacy.profile_snapshot is None
    with pytest.raises(ModelNotFoundError):
        recover_profile_from_snapshot(loaded_legacy)

    # 3. Registered profile precedence: registry resolves registered model even if snapshot is absent or corrupt
    registry = ModelRegistry()
    registered_profile = ModelProfile.for_container(
        id="registered/llama-3-8b",
        image="ghcr.io/org/llama:latest",
    )
    registry.register(registered_profile)

    # Precedence logic as implemented in InferWeave._resolve_profile:
    def resolve_profile(record: DeploymentRecord) -> ModelProfile:
        try:
            return registry.get(record.model)
        except ModelNotFoundError:
            return recover_profile_from_snapshot(record)

    # Legacy record resolves from registry
    assert resolve_profile(loaded_legacy) == registered_profile

    # If registry has a different/updated definition for the model, registry takes precedence
    override_profile = ModelProfile.for_container(
        id="custom/valid-recovered-model",
        image="ghcr.io/org/valid:v2-updated",
    )
    registry.register(override_profile)
    assert resolve_profile(loaded) == override_profile
