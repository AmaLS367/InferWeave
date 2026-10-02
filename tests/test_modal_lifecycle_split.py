"""Modal scale-to-zero (scaledown window) is independent of InferWeave's full-destroy timer."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from support import AUDIO_MODEL, deploy_on_modal, make_weave, patched_modal

from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.domain.options import DeploymentOptions
from inferweave.models.enums import DeploymentState
from inferweave.providers.modal_provider import DEFAULT_SCALEDOWN_WINDOW_SECONDS


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "deployments.db"


async def _scaledown_for(weave, **deploy_kwargs) -> int | None:
    captured: dict = {}
    real_function = __import__("modal").App.function

    def spy(self, *args, **kwargs):
        captured.update(kwargs)
        return real_function(self, *args, **kwargs)

    with patch("modal.App.function", spy):
        await deploy_on_modal(weave, **deploy_kwargs)
    return captured.get("scaledown_window")


@pytest.mark.asyncio
async def test_scaledown_window_is_independent_of_destroy_timer(db_path: Path):
    weave = make_weave(db_path)
    scaledown = await _scaledown_for(
        weave, destroy_after_idle_mins=1440, scaledown_window_seconds=600
    )
    assert scaledown == 600  # not 1440 * 60

    record = (await weave.list_records())[0]
    assert record.options.autostop.idle_minutes == 1440
    assert record.options.provider.scaledown_window_seconds == 600


@pytest.mark.asyncio
async def test_scaledown_default_does_not_follow_destroy_timer(db_path: Path):
    weave = make_weave(db_path)
    assert await _scaledown_for(weave, destroy_after_idle_mins=5) == DEFAULT_SCALEDOWN_WINDOW_SECONDS
    assert await _scaledown_for(weave, autostop_mins=15) == DEFAULT_SCALEDOWN_WINDOW_SECONDS
    # legacy custom_args spelling keeps working
    assert (
        await _scaledown_for(weave, custom_args={"scaledown_window": 90}, autostop_mins=15) == 90
    )


@pytest.mark.asyncio
async def test_explicit_parameter_overrides_custom_arg(db_path: Path):
    weave = make_weave(db_path)
    scaledown = await _scaledown_for(
        weave, custom_args={"scaledown_window_seconds": 90}, scaledown_window_seconds=240
    )
    assert scaledown == 240


@pytest.mark.asyncio
async def test_legacy_autostop_mins_still_sets_the_destroy_timer(db_path: Path):
    weave = make_weave(db_path)
    legacy = await deploy_on_modal(weave, autostop_mins=15)
    assert legacy.autostop_mins == 15
    modern = await deploy_on_modal(weave, destroy_after_idle_mins=15)
    assert modern.autostop_mins == 15
    default = await deploy_on_modal(weave)
    assert default.autostop_mins == 30  # historical default preserved
    disabled = await deploy_on_modal(weave, destroy_after_idle_mins=None)
    assert disabled.autostop_mins is None
    zero = await deploy_on_modal(weave, destroy_after_idle_mins=0)
    assert zero.autostop_mins is None

    watchdog = weave.lifecycle_service._watchdog.scheduled_checks  # type: ignore[attr-defined]
    assert legacy.id in watchdog and modern.id in watchdog
    assert disabled.id not in watchdog and zero.id not in watchdog


@pytest.mark.asyncio
async def test_conflicting_legacy_and_new_names_are_rejected(db_path: Path):
    weave = make_weave(db_path)
    with pytest.raises(ValueError, match="legacy alias"):
        await weave.deploy(
            model=AUDIO_MODEL, provider="modal", autostop_mins=10, destroy_after_idle_mins=20
        )
    # identical values are harmless
    await deploy_on_modal(weave, autostop_mins=10, destroy_after_idle_mins=10)


def test_destroy_after_idle_mins_custom_arg_spelling_for_cli_users():
    opts = DeploymentOptions.from_custom_args(
        {"destroy_after_idle_mins": 1440, "scaledown_window_seconds": 300}, autostop_mins=30
    )
    assert opts.autostop.idle_minutes == 1440
    assert opts.provider.scaledown_window_seconds == 300
    assert DeploymentOptions.from_custom_args({"destroy_after_idle_mins": 0}).autostop.enabled is False


@pytest.mark.asyncio
async def test_scale_to_zero_does_not_mark_the_deployment_stopped(db_path: Path):
    """An app whose containers scaled to zero still exists: it must not look STOPPED."""
    probe = MockHealthcheckProbeAdapter(default_healthy=False)  # cold: probes time out
    weave = make_weave(db_path, probe=probe)
    dep = await deploy_on_modal(weave, destroy_after_idle_mins=1440, scaledown_window_seconds=60)
    with patched_modal(DeploymentState.STARTING) as mocks:  # app exists, no containers
        status = await dep.refresh()
    assert status.state != DeploymentState.STOPPED
    assert status.state in {DeploymentState.STARTING, DeploymentState.UNHEALTHY}
    record = await weave.lifecycle_service.get_record(dep.id)
    assert record is not None and record.state != DeploymentState.STOPPED
    assert record.stopped_at is None
    mocks.stop_app.assert_not_awaited()
    assert record.is_attachable()

    # ... and the endpoint is still logical endpoint once the app wakes up
    probe._default_healthy = True
    with patched_modal(DeploymentState.STARTING):
        assert (await dep.refresh()).state == DeploymentState.HEALTHY
    assert dep.endpoint_url == status.endpoint_url


@pytest.mark.asyncio
async def test_idle_past_scaledown_but_before_destroy_window_does_not_destroy(db_path: Path):
    weave = make_weave(db_path)
    dep = await deploy_on_modal(weave, destroy_after_idle_mins=1440, scaledown_window_seconds=600)
    t0 = datetime.now(UTC)
    await weave.lifecycle_service.touch_activity(dep.id, now=t0)

    with patched_modal() as mocks:
        # 2 hours idle: far beyond the 10 minute scale-to-zero window, far below 24h destroy
        assert await weave.lifecycle_service.check_and_autostop(
            dep.id, now=t0 + timedelta(hours=2)
        ) is False
    mocks.stop_app.assert_not_awaited()
    assert dep.state != DeploymentState.STOPPED
    assert dep.is_idle() is False


@pytest.mark.asyncio
async def test_full_destroy_still_stops_the_modal_app_after_long_idle(db_path: Path):
    weave = make_weave(db_path)
    dep = await deploy_on_modal(weave, destroy_after_idle_mins=1440, scaledown_window_seconds=600)
    t0 = datetime.now(UTC)
    await weave.lifecycle_service.touch_activity(dep.id, now=t0)

    with patched_modal() as mocks:
        assert await weave.lifecycle_service.check_and_autostop(
            dep.id, now=t0 + timedelta(hours=24, minutes=1)
        ) is True
    mocks.stop_app.assert_awaited_once_with(dep.id)
    assert dep.state == DeploymentState.STOPPED
    record = await weave.lifecycle_service.get_record(dep.id)
    assert record is not None and record.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_activity_resets_the_long_idle_destroy_timer(db_path: Path):
    weave = make_weave(db_path)
    dep = await deploy_on_modal(weave, destroy_after_idle_mins=1440)
    t0 = datetime.now(UTC)
    await weave.lifecycle_service.touch_activity(dep.id, now=t0)
    # activity 23h later pushes the destroy deadline out by another 24h
    await weave.lifecycle_service.touch_activity(dep.id, now=t0 + timedelta(hours=23))

    with patched_modal() as mocks:
        assert await weave.lifecycle_service.check_and_autostop(
            dep.id, now=t0 + timedelta(hours=25)
        ) is False
        mocks.stop_app.assert_not_awaited()
        assert await weave.lifecycle_service.check_and_autostop(
            dep.id, now=t0 + timedelta(hours=47, minutes=1)
        ) is True
    mocks.stop_app.assert_awaited_once()


@pytest.mark.asyncio
async def test_persisted_activity_survives_a_fresh_inferweave_instance(db_path: Path):
    weave_a = make_weave(db_path)
    dep = await deploy_on_modal(weave_a, destroy_after_idle_mins=1440)
    active_at = datetime.now(UTC) - timedelta(hours=20)
    await weave_a.lifecycle_service.touch_activity(dep.id, now=active_at)

    weave_b = make_weave(db_path)
    attached = await weave_b.attach(dep.id)
    # The restart did NOT reset the 24h timer: 20h of idleness are remembered.
    assert attached.last_activity_at == active_at
    with patched_modal() as mocks:
        assert await weave_b.lifecycle_service.check_and_autostop(
            dep.id, now=active_at + timedelta(hours=23)
        ) is False
        assert await weave_b.lifecycle_service.check_and_autostop(
            dep.id, now=active_at + timedelta(hours=24, minutes=1)
        ) is True
    mocks.stop_app.assert_awaited_once_with(dep.id)


@pytest.mark.asyncio
async def test_activity_persistence_is_throttled(db_path: Path):
    weave = make_weave(db_path)
    weave.lifecycle_service._activity_persist_interval = 60.0
    dep = await deploy_on_modal(weave)
    repo = weave.lifecycle_service.repository
    first = (await repo.get(dep.id)).last_activity_at
    t0 = datetime.now(UTC)

    await weave.lifecycle_service.touch_activity(dep.id, now=t0 + timedelta(seconds=10))
    assert (await repo.get(dep.id)).last_activity_at == first  # within window: memory only
    assert dep.last_activity_at == t0 + timedelta(seconds=10)  # in-memory timer always exact
    await weave.lifecycle_service.touch_activity(dep.id, now=t0 + timedelta(seconds=120))
    assert (await repo.get(dep.id)).last_activity_at == t0 + timedelta(seconds=120)


@pytest.mark.asyncio
async def test_explicit_stop_is_a_real_termination(db_path: Path):
    weave = make_weave(db_path)
    dep = await deploy_on_modal(weave, scaledown_window_seconds=60)
    with patched_modal() as mocks:
        await dep.stop()
    mocks.stop_app.assert_awaited_once_with(dep.id)
    assert dep.state == DeploymentState.STOPPED
    record = await weave.lifecycle_service.get_record(dep.id)
    assert record is not None and record.state == DeploymentState.STOPPED
    assert record.stopped_at is not None


@pytest.mark.asyncio
async def test_registration_preserves_provider_persisted_record(db_path: Path):
    """register_deployment must update, not replace, the record the provider wrote."""
    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )
    from inferweave.providers.modal_provider import ModalProvider
    from inferweave.providers.router import ProviderRouter

    weave = make_weave(db_path)
    router: ProviderRouter = weave.router
    router.register(ModalProvider(repository=SqliteDeploymentRepository(db_path)))
    dep = await deploy_on_modal(weave, destroy_after_idle_mins=120, scaledown_window_seconds=45)
    record = await weave.lifecycle_service.get_record(dep.id)
    assert record is not None
    assert record.options.provider.scaledown_window_seconds == 45
    assert record.options.autostop.idle_minutes == 120
    assert record.workload_type is not None
    assert record.last_activity_at is not None
