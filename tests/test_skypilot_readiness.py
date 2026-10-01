"""Regression tests: SkyPilot deployments are only HEALTHY after a verified readiness probe.

Mirrors tests/test_modal_readiness.py. A launched / UP cluster proves infrastructure only.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from inferweave import DeploymentState, InferWeave
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.memory_repository import InMemoryDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.core.exceptions import HealthcheckTimeoutError
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.healthcheck import ProbeOutcome, ProbeResult
from inferweave.models.profile import HealthcheckConfig
from inferweave.providers.router import ProviderRouter
from inferweave.providers.skypilot import SkyPilotProvider
from inferweave.services.healthcheck_service import HealthcheckService
from inferweave.services.lifecycle_service import LifecycleService

ENDPOINT = "http://1.2.3.4:8000"
MODEL = "fish-s2-pro"


class _StateRecordingRepository(InMemoryDeploymentRepository):
    """In-memory repository that records every persisted state transition."""

    def __init__(self) -> None:
        super().__init__()
        self.saved_states: list[DeploymentState] = []

    async def save(self, record: DeploymentRecord) -> None:
        self.saved_states.append(record.state)
        await super().save(record)


def _build_weave(
    probe_adapter: MockHealthcheckProbeAdapter,
    provider_repo: InMemoryDeploymentRepository | None = None,
) -> tuple[InferWeave, InMemoryDeploymentRepository]:
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    lifecycle_repo = InMemoryDeploymentRepository()
    lifecycle_svc = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        repository=lifecycle_repo,
        healthcheck_service=healthcheck_svc,
    )
    router = ProviderRouter()
    if provider_repo is not None:
        router.register(SkyPilotProvider("runpod", repository=provider_repo))
    weave = InferWeave(
        router=router,
        healthcheck_service=healthcheck_svc,
        lifecycle_service=lifecycle_svc,
    )
    _fast_healthcheck(weave)  # no 10s initial readiness delay in unit tests
    return weave, lifecycle_repo


def _fast_healthcheck(weave: InferWeave, **overrides: object) -> None:
    profile = weave.registry.get(MODEL).model_copy(deep=True)
    profile.healthcheck = HealthcheckConfig(
        initial_delay_seconds=0,
        timeout_seconds=0.1,
        probe_interval_seconds=0.02,
        **overrides,
    )
    weave.register_model(profile)


def _mock_sky(cluster_status: str = "UP") -> MagicMock:
    sky = MagicMock()
    sky.launch = MagicMock(return_value=(1, None))
    sky.endpoints = MagicMock(return_value={8000: ENDPOINT})
    cluster = MagicMock()
    cluster.status = cluster_status
    sky.status = MagicMock(return_value=[cluster])
    return sky


@contextmanager
def _patched_skypilot(sky: MagicMock | None = None) -> Iterator[MagicMock]:
    sky = sky or _mock_sky()
    with (
        patch.object(SkyPilotProvider, "_ensure_supported_platform", return_value=None),
        patch.object(SkyPilotProvider, "_get_sky_module", return_value=sky),
    ):
        yield sky


def _hc(weave: InferWeave) -> HealthcheckConfig:
    return weave.registry.get(MODEL).healthcheck


async def _deploy(weave: InferWeave, **kwargs: object):
    return await weave.deploy(model=MODEL, provider="runpod", gpu_type="L4", **kwargs)


@pytest.mark.asyncio
async def test_skypilot_provider_never_persists_healthy_on_deploy():
    """SkyPilotProvider.deploy() alone transitions PROVISIONING -> STARTING, never HEALTHY."""
    repo = _StateRecordingRepository()
    weave, _ = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True), repo)

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=False)

    assert repo.saved_states == [DeploymentState.PROVISIONING, DeploymentState.STARTING]
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.STARTING
    assert record.endpoint_url == ENDPOINT
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_skypilot_no_endpoint_stays_provisioning():
    sky = _mock_sky()
    sky.endpoints = MagicMock(return_value={})
    weave, lifecycle_repo = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True))

    with _patched_skypilot(sky):
        deployment = await _deploy(weave, wait_for_ready=False)

    assert deployment.state == DeploymentState.PROVISIONING
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.PROVISIONING


@pytest.mark.asyncio
async def test_skypilot_no_wait_has_endpoint_but_is_not_healthy():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=False)

    assert probe_adapter.probe_call_count == 0
    assert deployment.endpoint_url == ENDPOINT
    assert deployment.state == DeploymentState.STARTING
    assert deployment.is_healthy is False
    assert deployment.status.ready_at is None

    # Persisted state agrees with the in-memory deployment.
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.state == deployment.state == DeploymentState.STARTING
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_skypilot_manual_readiness_transitions_memory_and_record_to_healthy():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=False)

    status = await deployment.wait_for_ready()

    assert probe_adapter.probe_call_count >= 1
    assert status.state == deployment.state == DeploymentState.HEALTHY
    assert deployment.status.ready_at is not None
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.HEALTHY
    assert record.ready_at is not None


@pytest.mark.asyncio
async def test_skypilot_wait_for_ready_returns_healthy_after_successful_probe():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=True)

    assert probe_adapter.probe_call_count >= 1
    assert deployment.state == DeploymentState.HEALTHY
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.HEALTHY
    assert record.ready_at is not None


@pytest.mark.asyncio
async def test_skypilot_readiness_timeout_marks_memory_and_record_failed():
    failing = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="container crashed",
    )
    probe_adapter = MockHealthcheckProbeAdapter(canned_results=[failing] * 50)
    weave, lifecycle_repo = _build_weave(probe_adapter)
    _fast_healthcheck(weave)

    with _patched_skypilot(), pytest.raises(HealthcheckTimeoutError):
        await _deploy(weave, wait_for_ready=True)

    (tracked,) = weave.list_deployments()
    assert tracked.state == DeploymentState.FAILED
    record = await lifecycle_repo.get(tracked.id)
    assert record is not None
    assert record.state == DeploymentState.FAILED
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_skypilot_disabled_healthcheck_does_not_fake_health():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)
    _fast_healthcheck(weave, enabled=False)

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=True)

    assert probe_adapter.probe_call_count == 0
    assert deployment.state == DeploymentState.STARTING
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_skypilot_dry_run_is_distinguishable_from_live_healthy():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)

    deployment = await _deploy(weave, dry_run=True, wait_for_ready=True)

    assert probe_adapter.probe_call_count == 0
    assert deployment.state == DeploymentState.PROVISIONING
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.is_dry_run is True
    assert record.state == DeploymentState.PROVISIONING


# --- Provider get_status: infrastructure state only -------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cluster_status", "expected"),
    [
        ("UP", DeploymentState.STARTING),
        ("ClusterStatus.UP", DeploymentState.STARTING),
        ("INIT", DeploymentState.PROVISIONING),
        ("STOPPED", DeploymentState.STOPPED),
    ],
)
async def test_skypilot_get_status_reports_infrastructure_state_never_healthy(
    cluster_status, expected
):
    provider = SkyPilotProvider("runpod")
    with _patched_skypilot(_mock_sky(cluster_status)):
        status = await provider.get_status("iw-test-cluster")

    assert status.state == expected
    assert status.state != DeploymentState.HEALTHY


@pytest.mark.asyncio
async def test_skypilot_get_status_up_cluster_exposes_endpoint():
    provider = SkyPilotProvider("runpod")
    with _patched_skypilot():
        status = await provider.get_status("iw-test-cluster")

    assert status.endpoint_url == ENDPOINT


@pytest.mark.asyncio
async def test_skypilot_get_status_missing_cluster_is_stopped():
    sky = _mock_sky()
    sky.status = MagicMock(return_value=[])
    provider = SkyPilotProvider("runpod")
    with _patched_skypilot(sky):
        status = await provider.get_status("iw-test-cluster")

    assert status.state == DeploymentState.STOPPED


# --- refresh_status: probe=False cannot create health, probe=True can -------------


@pytest.mark.asyncio
async def test_refresh_status_without_probe_cannot_create_false_health():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=False)
        status = await weave.lifecycle_service.refresh_status(deployment.id, probe=False)

    assert probe_adapter.probe_call_count == 0
    assert status.state == DeploymentState.STARTING
    assert deployment.state == DeploymentState.STARTING
    assert deployment.is_healthy is False
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.STARTING
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_refresh_status_with_probe_promotes_to_healthy_after_real_probe():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=False)
        assert probe_adapter.probe_call_count == 0
        status = await weave.lifecycle_service.refresh_status(
            deployment.id, probe=True, healthcheck_config=_hc(weave)
        )

    assert probe_adapter.probe_call_count == 1
    assert status.state == DeploymentState.HEALTHY
    assert deployment.state == DeploymentState.HEALTHY
    assert status.ready_at is not None
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.HEALTHY
    assert record.ready_at is not None


@pytest.mark.asyncio
async def test_refresh_status_with_failing_probe_stays_starting():
    not_ready = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="weights still loading",
    )
    probe_adapter = MockHealthcheckProbeAdapter(canned_results=[not_ready])
    weave, lifecycle_repo = _build_weave(probe_adapter)

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=False)
        status = await weave.lifecycle_service.refresh_status(
            deployment.id, probe=True, healthcheck_config=_hc(weave)
        )

    assert probe_adapter.probe_call_count == 1
    assert status.state == DeploymentState.STARTING
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_refresh_status_without_probe_keeps_earlier_verified_health():
    """A state proven by a real probe is not downgraded by a probe-less infra refresh."""
    weave, lifecycle_repo = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True))

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=True)
        status = await weave.lifecycle_service.refresh_status(deployment.id, probe=False)

    assert status.state == DeploymentState.HEALTHY
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.HEALTHY


@pytest.mark.asyncio
async def test_refresh_status_propagates_stopped_infrastructure():
    weave, lifecycle_repo = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True))

    with _patched_skypilot():
        deployment = await _deploy(weave, wait_for_ready=True)
    assert deployment.state == DeploymentState.HEALTHY

    with _patched_skypilot(_mock_sky("STOPPED")):
        status = await weave.lifecycle_service.refresh_status(
            deployment.id, probe=True, healthcheck_config=_hc(weave)
        )

    assert status.state == DeploymentState.STOPPED
    assert deployment.state == DeploymentState.STOPPED
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STOPPED
