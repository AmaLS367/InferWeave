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
        healthcheck_service=HealthcheckService(probe_port=MockHealthcheckProbeAdapter(default_healthy=True)),
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
        stopped = await lifecycle_svc.check_and_autostop(deployment.id, now=simulated_now)
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

