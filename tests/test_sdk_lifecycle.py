"""Integration tests for Deployment lifecycle, autostop, and activity tracking via InferWeave SDK."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from inferweave import DeploymentState, InferWeave
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.services.healthcheck_service import HealthcheckService
from inferweave.services.lifecycle_service import LifecycleService


@pytest.mark.asyncio
async def test_sdk_deploy_wires_lifecycle_tracking():
    mock_watchdog = MockWatchdogAdapter()
    lifecycle_svc = LifecycleService(watchdog_port=mock_watchdog)
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)

    weave = InferWeave(
        healthcheck_service=healthcheck_svc,
        lifecycle_service=lifecycle_svc,
    )

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://test.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            autostop_mins=10,
            custom_args={"max_model_len": 2048},
            wait_for_ready=False,
        )

        assert deployment.autostop_mins == 10
        assert deployment.last_activity_at is not None
        assert deployment.is_idle() is False

        state = lifecycle_svc.get_state(deployment.id)
        assert state is not None
        assert state.policy.idle_minutes == 10
        assert state.policy.enabled is True


@pytest.mark.asyncio
async def test_sdk_deployment_activity_and_autostop_trigger():
    mock_watchdog = MockWatchdogAdapter()
    lifecycle_svc = LifecycleService(watchdog_port=mock_watchdog)
    weave = InferWeave(
        healthcheck_service=HealthcheckService(
            probe_port=MockHealthcheckProbeAdapter(default_healthy=True)
        ),
        lifecycle_service=lifecycle_svc,
    )

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://test.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            autostop_mins=5,
            wait_for_ready=False,
        )

        # Record activity resets last_activity_at
        t0 = deployment.last_activity_at
        deployment.record_activity()
        t1 = deployment.last_activity_at
        assert t1 is not None and t0 is not None and t1 >= t0

        # Simulate idle exceeding 5 minutes
        simulated_now = t1 + timedelta(minutes=6)
        assert lifecycle_svc.is_idle(deployment.id, now=simulated_now) is True

        # Trigger autostop
        stopped = await lifecycle_svc.check_and_autostop(
            deployment.id, now=simulated_now
        )
        assert stopped is True
        assert deployment.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_sdk_healthcheck_updates_last_activity():
    lifecycle_svc = LifecycleService(watchdog_port=MockWatchdogAdapter())
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)

    weave = InferWeave(
        healthcheck_service=healthcheck_svc,
        lifecycle_service=lifecycle_svc,
    )

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://test.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            autostop_mins=10,
            wait_for_ready=False,
        )

        past_time = datetime.now(UTC) - timedelta(minutes=2)
        lifecycle_svc.record_activity(deployment.id, now=past_time)
        assert deployment.last_activity_at == past_time

        # Run health probe -> should touch activity
        probe = await deployment.check_health()
        assert probe.is_healthy is True
        assert deployment.last_activity_at > past_time


@pytest.mark.asyncio
async def test_sdk_refresh_retains_model_and_reconciles_unhealthy():
    # Probe adapter returns unhealthy probe result
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=False)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    lifecycle_svc = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        healthcheck_service=healthcheck_svc,
    )

    weave = InferWeave(
        healthcheck_service=healthcheck_svc,
        lifecycle_service=lifecycle_svc,
    )

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://test.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=False,
        )

        assert deployment.model == "fish-s2-pro"
        assert deployment.state == DeploymentState.HEALTHY

        # Refresh status: probe will fail, so reconciled state should be UNHEALTHY
        # and model should STILL be "fish-s2-pro", NOT "unknown"!
        refreshed = await deployment.refresh()
        assert refreshed.model == "fish-s2-pro"
        assert refreshed.state == DeploymentState.UNHEALTHY
        assert deployment.state == DeploymentState.UNHEALTHY


@pytest.mark.asyncio
async def test_sdk_weave_stop_and_get_status():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    lifecycle_svc = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        healthcheck_service=healthcheck_svc,
    )

    weave = InferWeave(
        healthcheck_service=healthcheck_svc,
        lifecycle_service=lifecycle_svc,
    )

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://test.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=False,
        )

        # Stop via weave facade
        await weave.stop(deployment.id)
        assert deployment.state == DeploymentState.STOPPED

        status = await weave.get_status(deployment.id)
        assert status.state == DeploymentState.STOPPED
        assert status.model == "fish-s2-pro"

        # Verify repository record
        record = await lifecycle_svc.get_record(deployment.id)
        assert record is not None
        assert record.state == DeploymentState.STOPPED
        assert record.model == "fish-s2-pro"
        assert record.stopped_at is not None


@pytest.mark.asyncio
async def test_sdk_weave_stop_failure_does_not_mark_stopped():
    from unittest.mock import AsyncMock

    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    lifecycle_svc = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        healthcheck_service=healthcheck_svc,
    )

    weave = InferWeave(
        healthcheck_service=healthcheck_svc,
        lifecycle_service=lifecycle_svc,
    )

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://test.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=False,
        )

        assert deployment.state == DeploymentState.HEALTHY

        # Mock the modal provider's stop to raise RuntimeError (network/API error)
        modal_provider = weave.router.get("modal")
        with (
            patch.object(modal_provider, "stop", AsyncMock(side_effect=RuntimeError("Cloud API Network Failure"))),
            pytest.raises(RuntimeError, match="Cloud API Network Failure"),
        ):
            await weave.stop(deployment.id)



        # Ensure deployment and repository record are NOT marked STOPPED
        assert deployment.state != DeploymentState.STOPPED
        record = await lifecycle_svc.get_record(deployment.id)
        assert record is not None
        assert record.state != DeploymentState.STOPPED
        assert record.state == DeploymentState.HEALTHY
        assert record.stopped_at is None


@pytest.mark.asyncio
async def test_lifecycle_dry_run_cross_process_no_provider_calls(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    from inferweave.adapters.lifecycle.sqlite_repository import (
        SqliteDeploymentRepository,
    )
    from inferweave.domain.deployment_record import DeploymentRecord

    db_path = tmp_path / "test_lifecycle_dryrun.db"
    repo = SqliteDeploymentRepository(db_path=db_path)

    # 1. Simulate a dry-run deployment saved by a previous CLI process
    record = DeploymentRecord(
        id="iw-dryrun-mock-12345",
        model="fish-s2-pro",
        provider="modal",
        state=DeploymentState.PROVISIONING,
        endpoint_url="http://dryrun-iw-dryrun-mock-12345.cloud:8080",
        is_dry_run=True,
    )
    await repo.save(record)

    # 2. In a brand new LifecycleService instance, attach a mock provider resolver
    mock_provider = MagicMock()
    mock_provider.get_status = AsyncMock()
    mock_provider.stop = AsyncMock()

    lifecycle_svc = LifecycleService(
        repository=repo,
        provider_resolver=lambda _: mock_provider,
    )

    # 3. Calling refresh_status on dry-run should NOT invoke provider API
    status = await lifecycle_svc.refresh_status("iw-dryrun-mock-12345")
    mock_provider.get_status.assert_not_called()
    assert status.id == "iw-dryrun-mock-12345"
    assert status.model == "fish-s2-pro"
    assert status.provider == "modal"
    assert status.state == DeploymentState.PROVISIONING
    assert status.endpoint_url == "http://dryrun-iw-dryrun-mock-12345.cloud:8080"

    # 4. Calling stop_deployment on dry-run should NOT invoke provider API
    await lifecycle_svc.stop_deployment("iw-dryrun-mock-12345")
    mock_provider.stop.assert_not_called()

    # 5. Persisted record is now STOPPED
    updated_record = await repo.get("iw-dryrun-mock-12345")
    assert updated_record is not None
    assert updated_record.state == DeploymentState.STOPPED
    assert updated_record.stopped_at is not None


