"""Deployment recovery across process restarts: InferWeave.attach() and InferWeave.find()."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from support import (
    AUDIO_MODEL,
    ENDPOINT,
    IMAGE_MODEL,
    deploy_on_modal,
    make_weave,
    patched_modal,
)

from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.sqlite_repository import SqliteDeploymentRepository
from inferweave.core.exceptions import (
    AmbiguousDeploymentError,
    DeploymentNotActiveError,
    DeploymentNotFoundError,
    ModelNotFoundError,
)
from inferweave.models.enums import DeploymentState, WorkloadType


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "deployments.db"


async def _read_record(db_path: Path, deployment_id: str):
    return await SqliteDeploymentRepository(db_path).get(deployment_id)


@pytest.mark.asyncio
async def test_attach_after_restart_preserves_persisted_metadata(db_path: Path):
    # Process A deploys and is "killed".
    weave_a = make_weave(db_path)
    original = await deploy_on_modal(
        weave_a,
        destroy_after_idle_mins=1440,
        scaledown_window_seconds=600,
        custom_args={"cleanup_on_failure": True},
    )
    await original.wait_for_ready()
    persisted = await _read_record(db_path, original.id)
    assert persisted is not None
    assert persisted.state == DeploymentState.HEALTHY

    # Process B starts with an empty in-memory state.
    weave_b = make_weave(db_path)
    assert weave_b.list_deployments() == []
    attached = await weave_b.attach(original.id)

    assert attached is not original
    assert attached.id == original.id
    assert attached.model == AUDIO_MODEL
    assert attached.provider == "modal"
    assert attached.endpoint_url == ENDPOINT
    assert attached.state == DeploymentState.HEALTHY
    assert attached.workload_type == WorkloadType.AUDIO
    assert attached.status.created_at == persisted.created_at
    assert attached.status.ready_at == persisted.ready_at
    assert attached.autostop_mins == 1440

    # Attaching must not rewrite the persisted record.
    after = await _read_record(db_path, original.id)
    assert after is not None
    assert after.created_at == persisted.created_at
    assert after.ready_at == persisted.ready_at
    assert after.options == persisted.options
    assert after.options.autostop.idle_minutes == 1440
    assert after.options.provider.scaledown_window_seconds == 600
    assert after.options.cleanup_on_failure is True
    assert after.state == DeploymentState.HEALTHY
    assert weave_b.list_deployments() == [attached]


@pytest.mark.asyncio
async def test_attached_handle_refresh_stop_and_health_closures_work(db_path: Path):
    weave_a = make_weave(db_path)
    dep_a = await deploy_on_modal(weave_a)

    probe = MockHealthcheckProbeAdapter(default_healthy=True)
    weave_b = make_weave(db_path, probe=probe)
    attached = await weave_b.attach(dep_a.id)

    # check_health / wait_for_ready use the recovered closures and reach the probe
    result = await attached.check_health()
    assert result.is_healthy is True
    assert probe.probed_urls[-1] == f"{ENDPOINT}/v1/health"
    status = await attached.wait_for_ready()
    assert status.state == DeploymentState.HEALTHY
    assert (await _read_record(db_path, dep_a.id)).state == DeploymentState.HEALTHY

    # refresh() reconciles with the provider and the readiness probe
    with patched_modal(DeploymentState.STARTING) as mocks:
        refreshed = await attached.refresh()
    assert refreshed.state == DeploymentState.HEALTHY
    assert refreshed.model == AUDIO_MODEL
    assert refreshed.endpoint_url == ENDPOINT
    mocks.fetch_state.assert_awaited()

    # stop() really stops the provider app and persists STOPPED
    with patched_modal() as mocks:
        await attached.stop()
    mocks.stop_app.assert_awaited_once_with(dep_a.id)
    assert attached.state == DeploymentState.STOPPED
    stopped = await _read_record(db_path, dep_a.id)
    assert stopped is not None and stopped.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_attach_nonexistent_deployment_raises(db_path: Path):
    weave = make_weave(db_path)
    with pytest.raises(DeploymentNotFoundError):
        await weave.attach("iw-modal-does-not-exist")


@pytest.mark.asyncio
async def test_attach_stopped_deployment_is_rejected(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    with patched_modal():
        await dep.stop()

    weave_b = make_weave(db_path)
    with pytest.raises(DeploymentNotActiveError, match="stopped"):
        await weave_b.attach(dep.id)
    assert await weave_b.find(model=AUDIO_MODEL, provider="modal") is None


@pytest.mark.asyncio
async def test_attach_failed_and_dry_run_deployments_are_rejected(db_path: Path):
    weave_a = make_weave(db_path)
    failed = await deploy_on_modal(weave_a)
    repo = SqliteDeploymentRepository(db_path)
    rec = await repo.get(failed.id)
    rec.mark_failed("readiness timed out")
    await repo.save(rec)

    dry = await weave_a.deploy(model=AUDIO_MODEL, provider="modal", dry_run=True)

    weave_b = make_weave(db_path)
    with pytest.raises(DeploymentNotActiveError, match="failed"):
        await weave_b.attach(failed.id)
    with pytest.raises(DeploymentNotActiveError, match="dry-run"):
        await weave_b.attach(dry.id)


@pytest.mark.asyncio
async def test_attach_unhealthy_scaled_to_zero_deployment_is_allowed(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    repo = SqliteDeploymentRepository(db_path)
    rec = await repo.get(dep.id)
    rec.state = DeploymentState.UNHEALTHY  # cold app: probes time out until it wakes
    await repo.save(rec)

    attached = await make_weave(db_path).attach(dep.id)
    assert attached.state == DeploymentState.UNHEALTHY


@pytest.mark.asyncio
async def test_attach_with_unregistered_model_raises_model_not_found(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    repo = SqliteDeploymentRepository(db_path)
    rec = await repo.get(dep.id)
    rec.model = "custom/unregistered-model"
    await repo.save(rec)

    with pytest.raises(ModelNotFoundError):
        await make_weave(db_path).attach(dep.id)


@pytest.mark.asyncio
async def test_find_by_model_and_provider_returns_attached_handle(db_path: Path):
    weave_a = make_weave(db_path)
    audio = await deploy_on_modal(weave_a, model=AUDIO_MODEL)
    await deploy_on_modal(weave_a, model=IMAGE_MODEL)

    weave_b = make_weave(db_path)
    found = await weave_b.find(model=AUDIO_MODEL, provider="modal")
    assert found is not None
    assert found.id == audio.id
    assert found.workload_type == WorkloadType.AUDIO
    assert await weave_b.find(model="fish-s2-pro", provider="runpod") is None
    assert await weave_b.find(model="meta-llama/Meta-Llama-3-8B-Instruct") is None
    # find() reuses the handle that attach() already tracks
    assert await weave_b.attach(audio.id) is found
    assert await weave_b.find(model=AUDIO_MODEL, provider="MODAL") is found


@pytest.mark.asyncio
async def test_find_with_multiple_matches_raises_ambiguity_error(db_path: Path):
    weave_a = make_weave(db_path)
    first = await deploy_on_modal(weave_a)
    second = await deploy_on_modal(weave_a)

    weave_b = make_weave(db_path)
    with pytest.raises(AmbiguousDeploymentError) as excinfo:
        await weave_b.find(model=AUDIO_MODEL, provider="modal")
    assert set(excinfo.value.candidate_ids) == {first.id, second.id}
    assert first.id in str(excinfo.value)

    # Explicit attach resolves the ambiguity; stopping one makes find() unambiguous.
    with patched_modal():
        await (await weave_b.attach(first.id)).stop()
    found = await weave_b.find(model=AUDIO_MODEL, provider="modal")
    assert found is not None and found.id == second.id


@pytest.mark.asyncio
async def test_attach_returns_tracked_handle_from_deploying_process(db_path: Path):
    weave = make_weave(db_path)
    deployed = await deploy_on_modal(weave)
    assert await weave.attach(deployed.id) is deployed
    assert await weave.find(model=AUDIO_MODEL, provider="modal") is deployed


@pytest.mark.asyncio
async def test_attach_after_local_stop_rejects_stale_tracked_handle(db_path: Path):
    weave = make_weave(db_path)
    deployed = await deploy_on_modal(weave)
    with patched_modal():
        await deployed.stop()
    with pytest.raises(DeploymentNotActiveError):
        await weave.attach(deployed.id)


@pytest.mark.asyncio
async def test_legacy_record_without_activity_gets_fresh_timer_and_backfill(
    db_path: Path,
):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a)
    repo = SqliteDeploymentRepository(db_path)
    rec = await repo.get(dep.id)
    rec.last_activity_at = None  # record written by an older InferWeave version
    rec.workload_type = None
    await repo.save(rec)

    before = datetime.now(UTC)
    attached = await make_weave(db_path).attach(dep.id)
    assert attached.last_activity_at is not None
    assert attached.last_activity_at >= before
    assert attached.workload_type == WorkloadType.AUDIO  # taken from the model profile
    backfilled = await repo.get(dep.id)
    assert backfilled.last_activity_at == attached.last_activity_at


@pytest.mark.asyncio
async def test_attach_restores_idle_timer_and_destroy_policy(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a, destroy_after_idle_mins=60)
    long_ago = datetime.now(UTC) - timedelta(minutes=50)
    await weave_a.lifecycle_service.touch_activity(dep.id, now=long_ago)

    weave_b = make_weave(db_path)
    attached = await weave_b.attach(dep.id)
    assert attached.last_activity_at == long_ago
    assert attached.is_idle() is False
    scheduled = weave_b.lifecycle_service._watchdog.scheduled_checks  # type: ignore[attr-defined]
    assert dep.id in scheduled  # watchdog is re-armed for the recovered deployment

    # 50 minutes of idleness survived the restart; 15 more minutes crosses the 60 minute limit.
    with patched_modal() as mocks:
        assert await weave_b.lifecycle_service.check_and_autostop(
            dep.id, now=long_ago + timedelta(minutes=59)
        ) is False
        assert await weave_b.lifecycle_service.check_and_autostop(
            dep.id, now=long_ago + timedelta(minutes=61)
        ) is True
    mocks.stop_app.assert_awaited_once_with(dep.id)
    assert (await _read_record(db_path, dep.id)).state == DeploymentState.STOPPED
