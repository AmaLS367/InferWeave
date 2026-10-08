"""Regression coverage for identity reuse, late creates and competing recovery processes."""

import asyncio
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import ALL_ENV, FakeProvider, accounts_config, deploy, make_weave

from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.core.exceptions import AccountUnavailableError, ProviderOperationError
from inferweave.domain.deployment_record import DeploymentRecord, ResourceRef
from inferweave.models.enums import DeploymentState

pytestmark = pytest.mark.asyncio
PROVIDERS = ("modal", "lightning", "runpod", "vast")


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_parallel_deploy_restart_and_rotation_preserve_identity(tmp_path, provider, monkeypatch):
    monkeypatch.setattr("inferweave.services.routing_service.sys", SimpleNamespace(platform="linux"))
    env = dict(ALL_ENV)
    cloud = FakeProvider(provider_name=provider)
    config = accounts_config((provider,), state_dir=tmp_path / "accounts")
    db = tmp_path / "d.db"
    first = make_weave(db, config, env, [cloud])
    deployments = await asyncio.gather(*(deploy(first, provider) for _ in range(4)))
    assert sorted(d.account for d in deployments) == ["a", "a", "b", "b"]
    await first.close()
    second = make_weave(db, config, env, [cloud])
    owner_a = next(d for d in deployments if d.account == "a")
    handle = await second.attach(owner_a.id)
    await handle.refresh()
    old_calls = list(cloud.calls)
    # Reusing a configured name with different keys is not proof of cloud identity.
    key = next(k for k in env if k.startswith(f"IW_{provider.upper()}_A_") and
               k.endswith(("TOKEN_SECRET", "API_KEY")))
    original = env[key]
    env[key] = "SENTINEL-another-cloud-identity"
    third = make_weave(db, config, env, [cloud])
    for operation in (third.attach(owner_a.id), third.get_status(owner_a.id), third.stop(owner_a.id)):
        with pytest.raises(AccountUnavailableError):
            await operation
    with pytest.raises(AccountUnavailableError):
        second._auth_headers_for(handle)
    assert cloud.calls == old_calls
    # Other owners stay usable; new deployments can use the new credential generation.
    other = next(d for d in deployments if d.account == "b")
    await third.stop(other.id)
    fresh = await deploy(third, provider, account="a")
    assert fresh.account == "a"
    assert (await third.lifecycle_service.get_record(fresh.id)).owner_fingerprint != (
        await third.lifecycle_service.get_record(owner_a.id)
    ).owner_fingerprint
    await third.stop(fresh.id)
    env[key] = original
    await third.stop(owner_a.id)
    await second.close()
    await third.close()


async def test_active_provision_is_protected_beyond_age_threshold(tmp_path):
    cloud = FakeProvider()
    env = dict(ALL_ENV)
    db = tmp_path / "d.db"
    config = accounts_config()
    first, second = (make_weave(db, config, env, [cloud]) for _ in range(2))
    entered, release = asyncio.Event(), asyncio.Event()
    ids = []

    async def slow(record, account):
        ids.append(record.id)
        persisted = await first.lifecycle_service.get_record(record.id)
        persisted.created_at = datetime.now(UTC) - timedelta(days=3)
        await first.lifecycle_service.repository.save(persisted)
        entered.set()
        await release.wait()

    cloud.before_provision = slow
    task = asyncio.create_task(deploy(first))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert await second.reconcile(min_age_seconds=0) == []
        assert await second.reconcile(min_age_seconds=0, confirmed_settled=tuple(ids)) == []
        with pytest.raises(ProviderOperationError, match="active"):
            await second.stop(ids[0])
        assert not cloud.ops("stop", "exists")
    finally:
        release.set()
    handle = await task
    assert (await second.lifecycle_service.get_record(handle.id)).state == DeploymentState.STARTING
    await first.close()
    await second.close()


async def test_concurrent_reconciliation_claims_and_rereads_once(tmp_path):
    cloud = FakeProvider()
    db = tmp_path / "d.db"
    record = DeploymentRecord(
        id="abandoned", model="fish-s2-pro", provider="modal", account="a",
        resource=ResourceRef(name="abandoned"), state=DeploymentState.FAILED,
        needs_reconciliation=True,
    )
    repo = SqliteDeploymentRepository(db)
    await repo.save(record)
    cloud.adopt(record.id, "a")
    original = cloud.resource_exists

    async def slow(*args):
        await asyncio.sleep(0.05)
        return await original(*args)

    cloud.resource_exists = slow
    instances = [make_weave(db, accounts_config(), dict(ALL_ENV), [cloud]) for _ in range(2)]
    results = await asyncio.gather(*(w.reconcile(min_age_seconds=0) for w in instances))
    assert sum(len(r) for r in results) == 1
    assert len(cloud.ops("stop")) == 1
    assert not (await repo.get(record.id)).needs_reconciliation
    for instance in instances:
        await instance.close()


@pytest.mark.parametrize("exists", [True, False])
async def test_late_create_requires_explicit_settlement_confirmation(tmp_path, exists):
    cloud = FakeProvider()
    db = tmp_path / "d.db"
    weave = make_weave(db, accounts_config(), dict(ALL_ENV), [cloud])
    record = DeploymentRecord(
        id="late", model="fish-s2-pro", provider="modal", account="a",
        resource=ResourceRef(name="late"), state=DeploymentState.FAILED,
        needs_reconciliation=True, creation_may_continue=True,
        created_at=datetime.now(UTC) - timedelta(days=2),
    )
    await weave.lifecycle_service.repository.save(record)
    if exists:
        cloud.adopt(record.id, "a")
    assert await weave.reconcile(min_age_seconds=0) == []
    assert cloud.calls == []
    status = await weave.get_status(record.id)
    assert status.state == DeploymentState.FAILED
    stored = await weave.lifecycle_service.get_record(record.id)
    assert stored.creation_may_continue and stored.needs_reconciliation
    assert cloud.calls == []
    [settled] = await weave.reconcile(min_age_seconds=0, confirmed_settled=(record.id,))
    assert not settled.creation_may_continue and not settled.needs_reconciliation
    assert cloud.live == {}
    await weave.close()


async def test_process_lock_survives_age_and_is_released_on_process_exit(tmp_path: Path):
    db = tmp_path / "d.db"
    repo = SqliteDeploymentRepository(db)
    script = """
import sys
from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
repository = SqliteDeploymentRepository(sys.argv[1])
with repository.operation_lock('pending') as acquired:
    print(acquired, flush=True)
    sys.stdin.read()
"""
    child = await asyncio.to_thread(subprocess.Popen,
        [sys.executable, "-c", script, str(db)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert await asyncio.to_thread(child.stdout.readline) == "True\n"
        with repo.operation_lock("pending") as acquired:
            assert not acquired
    finally:
        await asyncio.to_thread(child.communicate, "", timeout=10)
    with repo.operation_lock("pending") as acquired:
        assert acquired


async def test_json_repository_serializes_writes_and_operations(tmp_path):
    from inferweave.adapters.lifecycle.json_repository import JsonDeploymentRepository

    path = tmp_path / "deployments.json"
    first, second = JsonDeploymentRepository(path), JsonDeploymentRepository(path)
    with first.operation_lock("pending") as acquired:
        assert acquired
        with second.operation_lock("pending") as concurrent:
            assert not concurrent
    records = [DeploymentRecord(id=str(i), model="fish-s2-pro", provider="modal") for i in range(20)]
    await asyncio.gather(*( (first if i % 2 else second).save(record) for i, record in enumerate(records)))
    assert len(await first.list_all()) == len(records)
