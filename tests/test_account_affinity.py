"""Deployments stay bound to the account that created them, across processes and rotations."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes import (
    MODAL_ENV,
    MODEL,
    FakeProvider,
    accounts_config,
    crash,
    deploy,
    fail,
    make_weave,
)

from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.core.exceptions import (
    AccountUnavailableError,
    ProvisioningUncertainError,
)
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord, ResourceRef
from inferweave.models.enums import DeploymentState

pytestmark = pytest.mark.asyncio


@pytest.fixture
def env() -> dict[str, str]:
    return dict(MODAL_ENV)


@pytest.fixture
def cloud() -> FakeProvider:
    """One fake Modal cloud shared by every simulated process."""
    return FakeProvider()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "deployments.db"


def process(db: Path, cloud: FakeProvider, env: dict[str, str], **kwargs):
    """A fresh InferWeave process (new AccountManager, no in-memory state) over ``db``."""
    return make_weave(db, accounts_config(), env, [cloud], **kwargs)


def owned_calls_ok(cloud: FakeProvider) -> None:
    assert not cloud.violations
    for call in cloud.ops("status", "stop", "exists"):
        assert cloud.owners[call.deployment_id] == call.account_id, call


async def test_restart_keeps_every_operation_on_owner(db, cloud, env) -> None:
    first = process(db, cloud, env)
    d_a = await deploy(first)
    d_b = await deploy(first)
    assert (d_a.account, d_b.account) == ("a", "b")
    await first.close()

    second = process(db, cloud, env)
    attached = await second.attach(d_b.id)
    assert attached.account == "b"
    status = await second.get_status(d_a.id)
    assert (status.account, status.state) == ("a", DeploymentState.HEALTHY)
    refreshed = await attached.refresh()
    assert refreshed.account == "b"
    await second.stop(d_b.id)
    found = await second.find(model=MODEL, provider="modal")
    assert found is not None and found.id == d_a.id and found.account == "a"
    await found.stop()
    assert cloud.live == {}
    assert [(c.op, c.account_id) for c in cloud.ops("status", "stop")] == [
        ("status", "a"),
        ("status", "b"),
        ("stop", "b"),
        ("stop", "a"),
    ]
    owned_calls_ok(cloud)
    await second.close()


async def test_endpoint_headers_follow_owner_after_restart(db, cloud, env) -> None:
    first = process(db, cloud, env)
    d_a, d_b = await deploy(first), await deploy(first)
    await first.close()
    second = process(db, cloud, env)
    a, b = await second.attach(d_a.id), await second.attach(d_b.id)
    assert second._auth_headers_for(a)["Modal-Key"] == env["IW_MODAL_A_PROXY_ID"]
    assert second._auth_headers_for(b)["Modal-Secret"] == env["IW_MODAL_B_PROXY_SECRET"]


async def test_autostop_watchdog_stops_under_owner_after_restart(
    db, cloud, env
) -> None:
    first = process(db, cloud, env)
    await deploy(first)
    d_b = await deploy(first, destroy_after_idle_mins=1)
    assert d_b.account == "b"
    await first.close()

    # Last activity long ago: the restarted watchdog must fire immediately.
    repo = SqliteDeploymentRepository(db)
    record = await repo.get(d_b.id)
    record.last_activity_at = datetime.now(UTC) - timedelta(minutes=10)
    await repo.save(record)

    watchdog = MockWatchdogAdapter()
    second = process(db, cloud, env, watchdog=watchdog)
    await second.attach(d_b.id)
    assert d_b.id in watchdog.scheduled_checks
    await watchdog.trigger_check(d_b.id)
    assert [(c.op, c.deployment_id, c.account_id) for c in cloud.ops("stop")] == [
        ("stop", d_b.id, "b")
    ]
    assert (await repo.get(d_b.id)).state == DeploymentState.STOPPED
    owned_calls_ok(cloud)


async def test_check_and_autostop_uses_owner(db, cloud, env) -> None:
    weave = process(db, cloud, env)
    await deploy(weave, account="b")
    d_a = await deploy(weave, account="a", destroy_after_idle_mins=1)
    later = datetime.now(UTC) + timedelta(minutes=5)
    assert await weave.lifecycle_service.check_and_autostop(d_a.id, now=later) is True
    assert cloud.accounts_for("stop") == ["a"]


async def test_concurrent_deploys_do_not_mix_accounts(db, cloud, env) -> None:
    weave = process(db, cloud, env)
    deployments = await asyncio.gather(*(deploy(weave) for _ in range(8)))
    assert sorted(d.account for d in deployments) == ["a"] * 4 + ["b"] * 4
    for d in deployments:
        assert cloud.owners[d.id] == d.account
        assert (await weave.lifecycle_service.get_record(d.id)).account == d.account
    statuses = await asyncio.gather(*(weave.get_status(d.id) for d in deployments))
    assert [s.account for s in statuses] == [d.account for d in deployments]
    await asyncio.gather(*(weave.stop(d.id) for d in deployments))
    assert cloud.live == {}
    owned_calls_ok(cloud)


async def test_endpoint_rotation_is_used_for_later_operations(db, cloud, env) -> None:
    weave = process(db, cloud, env)
    d = await deploy(weave)
    assert d.account == "a"
    env["IW_MODAL_A_PROXY_ID"] = "SENTINEL-modal-a-rotated-proxy"
    await weave.get_status(d.id)
    assert (
        cloud.accounts_seen[-1].secret("token_secret")
        == env["IW_MODAL_A_TOKEN_SECRET"]
    )
    assert weave._auth_headers_for(d)["Modal-Key"] == "SENTINEL-modal-a-rotated-proxy"

    restarted = process(db, cloud, env)
    await restarted.stop(d.id)
    stop_account = cloud.accounts_seen[-1]
    assert (stop_account.id, stop_account.secret("token_secret")) == (
        "a",
        env["IW_MODAL_A_TOKEN_SECRET"],
    )
    owned_calls_ok(cloud)


async def test_removed_account_blocks_operations_without_substitute(
    db, cloud, env
) -> None:
    first = process(db, cloud, env)
    await deploy(first)
    d_b = await deploy(first)
    assert d_b.account == "b"
    await first.close()
    calls_before = list(cloud.calls)

    del env["IW_MODAL_B_TOKEN_ID"]
    second = process(db, cloud, env)
    with pytest.raises(AccountUnavailableError, match="'b'"):
        await second.attach(d_b.id)
    with pytest.raises(AccountUnavailableError):
        await second.get_status(d_b.id)
    with pytest.raises(AccountUnavailableError):
        await second.stop(d_b.id)
    assert cloud.calls == calls_before  # no provider call at all, under any account
    record = await second.lifecycle_service.get_record(d_b.id)
    assert record.account == "b" and record.state != DeploymentState.STOPPED
    assert cloud.live[d_b.id] == "b"


async def test_legacy_record_without_account_loads_as_ambient(db, cloud, env) -> None:
    repo = SqliteDeploymentRepository(db)
    legacy = DeploymentRecord(
        id="iw-legacy-1",
        model=MODEL,
        provider="modal",
        state=DeploymentState.STARTING,
        endpoint_url="https://iw-legacy-1.modal.run",
    )
    await repo.save(legacy)
    with sqlite3.connect(db) as conn:
        (raw,) = conn.execute(
            "SELECT data_json FROM deployments WHERE id='iw-legacy-1'"
        ).fetchone()
        data = json.loads(raw)
        for key in ("account", "resource", "needs_reconciliation"):
            data.pop(key, None)
        conn.execute(
            "UPDATE deployments SET data_json=? WHERE id='iw-legacy-1'",
            (json.dumps(data),),
        )
    loaded = await repo.get("iw-legacy-1")
    assert loaded.account == "ambient" and loaded.resource is None
    assert loaded.needs_reconciliation is False

    cloud.adopt("iw-legacy-1", "ambient")
    weave = process(db, cloud, env)
    handle = await weave.attach("iw-legacy-1")
    assert handle.account == "ambient"
    status = await weave.get_status("iw-legacy-1")
    assert status.account == "ambient"
    await weave.stop("iw-legacy-1")
    assert [(c.op, c.account_id) for c in cloud.ops("status", "stop")] == [
        ("status", "ambient"),
        ("stop", "ambient"),
    ]
    # Ambient endpoint auth comes from the environment (conftest proxy tokens).
    assert weave._auth_headers_for(handle)["Modal-Key"] == "wk-test-proxy-id-0000"


async def test_status_of_stopped_deployment_keeps_owner(db, cloud, env) -> None:
    weave = process(db, cloud, env)
    await deploy(weave)
    d = await deploy(weave)
    await weave.stop(d.id)
    status = await weave.get_status(d.id)
    assert status.state == DeploymentState.STOPPED
    assert status.account == "b"
    assert d.account == "b"


# --- reconcile -----------------------------------------------------------------------------------


async def _uncertain_deploy(weave, cloud: FakeProvider, *, create: bool) -> str:
    cloud.script("provision", fail(FailureKind.TRANSIENT, create=create))
    cloud.script("exists", crash(ConnectionError("down")))
    with pytest.raises(ProvisioningUncertainError) as info:
        await deploy(weave)
    assert info.value.deployment_id is not None
    return info.value.deployment_id


async def test_reconcile_destroys_leftover_under_owner(db, cloud, env) -> None:
    weave = process(db, cloud, env)
    await deploy(weave)  # a
    leftover = await _uncertain_deploy(weave, cloud, create=True)  # b
    assert cloud.live[leftover] == "b"
    await weave.close()

    restarted = process(db, cloud, env)
    updated = await restarted.reconcile(min_age_seconds=0)
    assert [r.id for r in updated] == [leftover]
    record = await restarted.lifecycle_service.get_record(leftover)
    assert record.state == DeploymentState.STOPPED and not record.needs_reconciliation
    assert leftover not in cloud.live
    tail = [(c.op, c.deployment_id, c.account_id) for c in cloud.ops("exists", "stop")][
        -3:
    ]
    assert tail == [
        ("exists", leftover, "b"),
        ("stop", leftover, "b"),
        ("exists", leftover, "b"),
    ]
    owned_calls_ok(cloud)
    assert await restarted.reconcile(min_age_seconds=0) == []


async def test_reconcile_clears_flag_when_resource_absent(db, cloud, env) -> None:
    weave = process(db, cloud, env)
    dep_id = await _uncertain_deploy(weave, cloud, create=False)
    updated = await weave.reconcile(min_age_seconds=0)
    assert [r.id for r in updated] == [dep_id]
    record = await weave.lifecycle_service.get_record(dep_id)
    assert record.state == DeploymentState.FAILED and not record.needs_reconciliation
    assert cloud.ops("stop") == []


async def _flagged_provisioning(
    db: Path, account: str, age: timedelta
) -> DeploymentRecord:
    record = DeploymentRecord(
        id=f"iw-modal-inflight-{account}",
        model=MODEL,
        provider="modal",
        account=account,
        resource=ResourceRef(name=f"iw-modal-inflight-{account}"),
        state=DeploymentState.PROVISIONING,
        needs_reconciliation=True,
        created_at=datetime.now(UTC) - age,
    )
    await SqliteDeploymentRepository(db).save(record)
    return record


async def test_reconcile_skips_young_provisioning_records(db, cloud, env) -> None:
    young = await _flagged_provisioning(db, "a", timedelta(seconds=30))
    weave = process(db, cloud, env)
    assert await weave.reconcile(min_age_seconds=1800) == []
    assert cloud.calls == []
    assert (await weave.lifecycle_service.get_record(young.id)).needs_reconciliation

    # Once old enough it is checked under its owner and closed out.
    updated = await weave.reconcile(min_age_seconds=10)
    assert [r.id for r in updated] == [young.id]
    record = await weave.lifecycle_service.get_record(young.id)
    assert record.state == DeploymentState.FAILED and not record.needs_reconciliation
    assert [(c.op, c.account_id) for c in cloud.calls] == [("exists", "a")]


async def test_reconcile_keeps_flag_when_account_missing(db, cloud, env) -> None:
    old = await _flagged_provisioning(db, "b", timedelta(hours=2))
    cloud.adopt(old.id, "b")
    del env["IW_MODAL_B_TOKEN_SECRET"]
    weave = process(db, cloud, env)
    assert await weave.reconcile(min_age_seconds=0) == []
    assert cloud.calls == []
    record = await weave.lifecycle_service.get_record(old.id)
    assert record.needs_reconciliation and record.account == "b"
    assert cloud.live == {old.id: "b"}
