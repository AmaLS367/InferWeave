"""Unit tests for Healthcheck & Readiness Polling through InferWeave SDK with simulated probe adapters."""

from unittest.mock import patch

import pytest

from inferweave import DeploymentState, InferWeave
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.core.exceptions import HealthcheckTimeoutError
from inferweave.domain.healthcheck import ProbeOutcome, ProbeResult
from inferweave.models.profile import HealthcheckConfig
from inferweave.services.healthcheck_service import HealthcheckService


@pytest.mark.asyncio
async def test_sdk_deploy_with_wait_for_ready_success():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    weave = InferWeave(healthcheck_service=healthcheck_svc)

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://my-model.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=True,
        )

        assert probe_adapter.probe_call_count == 1
        assert "https://my-model.modal.run/v1/health" in probe_adapter.probed_urls
        assert deployment.state == DeploymentState.HEALTHY
        assert deployment.status.ready_at is not None
        assert deployment.is_healthy is True


@pytest.mark.asyncio
async def test_sdk_deploy_without_wait_for_ready_and_manual_wait():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    weave = InferWeave(healthcheck_service=healthcheck_svc)

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://async-model.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=False,
        )

        # Did not probe yet
        assert probe_adapter.probe_call_count == 0

        # Now manually wait for readiness
        status = await deployment.wait_for_ready()
        assert probe_adapter.probe_call_count == 1
        assert status.state == DeploymentState.HEALTHY
        assert status.ready_at is not None
        assert deployment.is_healthy is True


@pytest.mark.asyncio
async def test_sdk_deployment_check_health_method():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    weave = InferWeave(healthcheck_service=healthcheck_svc)

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://check-model.modal.run"),
    ):
        deployment = await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=False,
        )

        probe = await deployment.check_health()
        assert probe.is_healthy is True
        assert probe.outcome == ProbeOutcome.SUCCESS
        assert probe.status_code == 200


@pytest.mark.asyncio
async def test_sdk_deploy_timeout_transitions_to_failed():
    failing_probe = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="Model container crashed during weight loading",
    )
    probe_adapter = MockHealthcheckProbeAdapter(canned_results=[failing_probe] * 20)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    weave = InferWeave(healthcheck_service=healthcheck_svc)

    # Fast timeout override via custom profile
    profile = weave.registry.get("fish-s2-pro").model_copy(deep=True)
    profile.healthcheck = HealthcheckConfig(
        initial_delay_seconds=0,
        timeout_seconds=0.1,
        probe_interval_seconds=0.02,
    )
    weave.register_model(profile)

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://crashing-model.modal.run"),
        pytest.raises(HealthcheckTimeoutError) as exc_info,
    ):
        await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=True,
        )

    assert "timed out" in str(exc_info.value)

    # Check that deployment tracked in active_deployments reflects FAILED state
    tracked = weave.list_deployments()
    assert len(tracked) == 1
    assert tracked[0].state == DeploymentState.FAILED
    assert "timed out" in (tracked[0].status.error_message or "").lower()
    assert exc_info.value.total_probes >= 1
    assert exc_info.value.deployment_id == tracked[0].id

    # Check that repository persisted record reflects FAILED state with error message and endpoint
    record = await weave.lifecycle_service.get_record(tracked[0].id)
    assert record is not None
    assert record.state == DeploymentState.FAILED
    assert "timed out" in (record.error_message or "").lower()
    assert record.endpoint_url == "https://crashing-model.modal.run"


@pytest.mark.asyncio
async def test_sdk_deploy_timeout_with_cleanup_on_failure():
    """Validates that cleanup_on_failure=True automatically stops the deployment on timeout."""
    failing_probe = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="Connection refused",
    )
    probe_adapter = MockHealthcheckProbeAdapter(canned_results=[failing_probe] * 10)
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    weave = InferWeave(healthcheck_service=healthcheck_svc)

    profile = weave.registry.get("fish-s2-pro").model_copy(deep=True)
    profile.healthcheck = HealthcheckConfig(
        initial_delay_seconds=0,
        timeout_seconds=0.05,
        probe_interval_seconds=0.01,
    )
    weave.register_model(profile)

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://cleanup-test.modal.run"),
        patch(
            "inferweave.providers.modal_provider.ModalProvider._stop_modal_app",
            return_value=None,
        ) as mock_stop,
        pytest.raises(HealthcheckTimeoutError),
    ):
        await weave.deploy(
            model="fish-s2-pro",
            provider="modal",
            wait_for_ready=True,
            custom_args={"cleanup_on_failure": True},
        )

    # Stop must have been invoked automatically
    assert mock_stop.called


@pytest.mark.asyncio
async def test_sdk_dry_run_bypasses_healthcheck():
    probe_adapter = MockHealthcheckProbeAdapter()
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    weave = InferWeave(healthcheck_service=healthcheck_svc)

    deployment = await weave.deploy(
        model="fish-s2-pro",
        provider="modal",
        dry_run=True,
        wait_for_ready=True,
    )

    assert probe_adapter.probe_call_count == 0
    assert deployment.state == DeploymentState.PROVISIONING
