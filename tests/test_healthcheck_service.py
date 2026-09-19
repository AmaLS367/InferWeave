"""Unit tests for HealthcheckService application service."""

import pytest

from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.core.exceptions import HealthcheckError, HealthcheckTimeoutError
from inferweave.domain.healthcheck import (
    ProbeOutcome,
    ProbeResult,
    ReadinessReport,
    ReadinessState,
)
from inferweave.models.profile import HealthcheckConfig
from inferweave.services.healthcheck_service import HealthcheckService


def test_build_probe_url_standard():
    service = HealthcheckService(probe_port=MockHealthcheckProbeAdapter())
    config = HealthcheckConfig(path="/health", port=8000)

    # IP with port already
    assert (
        service.build_probe_url("http://1.2.3.4:8000", config)
        == "http://1.2.3.4:8000/health"
    )

    # Cloud HTTPS URL without port
    assert (
        service.build_probe_url("https://my-app.modal.run", config)
        == "https://my-app.modal.run/health"
    )

    # URL with trailing slash
    assert (
        service.build_probe_url("https://my-app.modal.run/", config)
        == "https://my-app.modal.run/health"
    )

    # Raw IP without scheme and without port
    assert service.build_probe_url("10.0.0.5", config) == "http://10.0.0.5:8000/health"


@pytest.mark.asyncio
async def test_wait_for_ready_disabled_config():
    service = HealthcheckService(probe_port=MockHealthcheckProbeAdapter())
    config = HealthcheckConfig(enabled=False)

    report = await service.wait_for_ready(
        endpoint_url="http://localhost:8000",
        config=config,
    )
    assert report.state == ReadinessState.READY
    assert report.total_probes == 0


@pytest.mark.asyncio
async def test_wait_for_ready_missing_endpoint():
    service = HealthcheckService(probe_port=MockHealthcheckProbeAdapter())
    config = HealthcheckConfig(enabled=True)

    with pytest.raises(HealthcheckError) as exc_info:
        await service.wait_for_ready(
            endpoint_url=None, config=config, deployment_id="dep-123"
        )

    assert "dep-123" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_for_ready_immediate_success():
    adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    service = HealthcheckService(probe_port=adapter)
    config = HealthcheckConfig(
        initial_delay_seconds=0,
        timeout_seconds=5,
        probe_interval_seconds=1,
    )

    report = await service.wait_for_ready(
        endpoint_url="http://localhost:8000",
        config=config,
        deployment_id="test-dep",
    )

    assert report.state == ReadinessState.READY
    assert report.total_probes == 1
    assert report.consecutive_successes == 1
    assert adapter.probe_call_count == 1


@pytest.mark.asyncio
async def test_wait_for_ready_success_after_retries():
    p_fail1 = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="Connection refused",
    )
    p_fail2 = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.HTTP_ERROR,
        status_code=503,
        error_message="Model loading",
    )
    p_success = ProbeResult(
        is_healthy=True,
        outcome=ProbeOutcome.SUCCESS,
        status_code=200,
    )

    adapter = MockHealthcheckProbeAdapter(canned_results=[p_fail1, p_fail2, p_success])
    service = HealthcheckService(probe_port=adapter)
    config = HealthcheckConfig(
        initial_delay_seconds=0,
        timeout_seconds=10,
        probe_interval_seconds=0,  # Fast interval for test
        consecutive_successes=1,
    )

    polls: list[ReadinessReport] = []

    def on_poll(r: ReadinessReport) -> None:
        polls.append(r)

    report = await service.wait_for_ready(
        endpoint_url="http://localhost:8000",
        config=config,
        on_poll=on_poll,
    )

    assert report.state == ReadinessState.READY
    assert report.total_probes == 3
    assert len(polls) == 3
    assert polls[0].state == ReadinessState.POLLING
    assert polls[1].state == ReadinessState.POLLING
    assert polls[2].state == ReadinessState.READY


@pytest.mark.asyncio
async def test_wait_for_ready_timeout_exceeded():
    p_fail = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="Host unreachable",
    )

    adapter = MockHealthcheckProbeAdapter(
        canned_results=[p_fail] * 10, default_healthy=False
    )
    service = HealthcheckService(probe_port=adapter)
    config = HealthcheckConfig(
        initial_delay_seconds=0,
        timeout_seconds=0.1,  # Fast timeout for test
        probe_interval_seconds=0.02,
    )

    with pytest.raises(HealthcheckTimeoutError) as exc_info:
        await service.wait_for_ready(
            endpoint_url="http://localhost:8000",
            config=config,
            deployment_id="dep-timeout",
        )

    err = exc_info.value
    assert err.deployment_id == "dep-timeout"
    assert "timed out after 0.1s" in str(err)
    assert err.total_probes >= 1


@pytest.mark.asyncio
async def test_check_health_single_probe():
    adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    service = HealthcheckService(probe_port=adapter)
    config = HealthcheckConfig(path="/v1/status")

    res = await service.check_health("http://localhost:8000", config)
    assert res.is_healthy is True
    assert adapter.probed_urls == ["http://localhost:8000/v1/status"]
