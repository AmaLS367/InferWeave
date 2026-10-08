"""Regression tests: Modal deployments are only HEALTHY after a verified readiness healthcheck."""

from unittest.mock import patch

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
from inferweave.services.healthcheck_service import HealthcheckService
from inferweave.services.lifecycle_service import LifecycleService

ENDPOINT = "https://readiness-test.modal.run"


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
    repository: InMemoryDeploymentRepository | None = None,
) -> tuple[InferWeave, InMemoryDeploymentRepository]:
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    lifecycle_repo = repository or InMemoryDeploymentRepository()
    lifecycle_svc = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        repository=lifecycle_repo,
        healthcheck_service=healthcheck_svc,
    )
    router = ProviderRouter()
    weave = InferWeave(
        router=router,
        healthcheck_service=healthcheck_svc,
        lifecycle_service=lifecycle_svc,
    )
    return weave, lifecycle_repo


def _fast_healthcheck(weave: InferWeave, **overrides: object) -> None:
    profile = weave.registry.get("fish-s2-pro").model_copy(deep=True)
    profile.healthcheck = HealthcheckConfig(
        initial_delay_seconds=0,
        timeout_seconds=0.1,
        probe_interval_seconds=0.02,
        **overrides,
    )
    weave.register_model(profile)


def _patched_modal_deploy():
    return (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value=ENDPOINT),
    )


@pytest.mark.asyncio
async def test_modal_provisioning_never_persists_healthy_on_deploy():
    """Provisioning transitions PROVISIONING -> STARTING (write-ahead first), never HEALTHY."""
    repo = _StateRecordingRepository()
    weave, _ = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True), repo)
    deploy_patch, url_patch = _patched_modal_deploy()

    with deploy_patch, url_patch:
        deployment = await weave.deploy(
            model="fish-s2-pro", provider="modal", wait_for_ready=False
        )

    assert repo.saved_states[0] == DeploymentState.PROVISIONING
    assert DeploymentState.STARTING in repo.saved_states
    assert DeploymentState.HEALTHY not in repo.saved_states
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.STARTING
    assert record.account == "ambient"
    assert record.needs_reconciliation is False
    assert record.endpoint_url == ENDPOINT
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_modal_no_wait_has_endpoint_but_is_not_healthy():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)
    deploy_patch, url_patch = _patched_modal_deploy()

    with deploy_patch, url_patch:
        deployment = await weave.deploy(
            model="fish-s2-pro", provider="modal", wait_for_ready=False
        )

    assert probe_adapter.probe_call_count == 0
    assert deployment.endpoint_url == ENDPOINT
    assert deployment.state == DeploymentState.STARTING
    assert deployment.state != DeploymentState.HEALTHY
    assert deployment.status.ready_at is None
    assert deployment.is_healthy is False

    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.STARTING
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_modal_manual_readiness_transitions_memory_and_record_to_healthy():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)
    deploy_patch, url_patch = _patched_modal_deploy()

    with deploy_patch, url_patch:
        deployment = await weave.deploy(
            model="fish-s2-pro", provider="modal", wait_for_ready=False
        )

    record = await lifecycle_repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STARTING

    status = await deployment.wait_for_ready()

    assert probe_adapter.probe_call_count >= 1
    assert status.state == DeploymentState.HEALTHY
    assert deployment.state == DeploymentState.HEALTHY
    assert deployment.status.ready_at is not None

    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.HEALTHY
    assert record.ready_at is not None


@pytest.mark.asyncio
async def test_modal_wait_for_ready_returns_healthy_after_successful_probe():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)
    deploy_patch, url_patch = _patched_modal_deploy()

    with deploy_patch, url_patch:
        deployment = await weave.deploy(
            model="fish-s2-pro", provider="modal", wait_for_ready=True
        )

    assert probe_adapter.probe_call_count >= 1
    assert deployment.state == DeploymentState.HEALTHY

    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.HEALTHY
    assert record.ready_at is not None


@pytest.mark.asyncio
async def test_modal_readiness_eventually_succeeds_after_failed_probes():
    not_ready = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="weights still loading",
    )
    ready = ProbeResult(is_healthy=True, outcome=ProbeOutcome.SUCCESS, status_code=200)
    probe_adapter = MockHealthcheckProbeAdapter(canned_results=[not_ready, not_ready, ready])
    weave, lifecycle_repo = _build_weave(probe_adapter)
    _fast_healthcheck(weave)
    deploy_patch, url_patch = _patched_modal_deploy()

    with deploy_patch, url_patch:
        deployment = await weave.deploy(
            model="fish-s2-pro", provider="modal", wait_for_ready=True
        )

    assert probe_adapter.probe_call_count == 3
    assert deployment.state == DeploymentState.HEALTHY
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.HEALTHY


@pytest.mark.asyncio
async def test_modal_readiness_timeout_marks_memory_and_record_failed():
    failing = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="container crashed",
    )
    probe_adapter = MockHealthcheckProbeAdapter(canned_results=[failing] * 50)
    weave, lifecycle_repo = _build_weave(probe_adapter)
    _fast_healthcheck(weave)
    deploy_patch, url_patch = _patched_modal_deploy()

    with deploy_patch, url_patch, pytest.raises(HealthcheckTimeoutError):
        await weave.deploy(model="fish-s2-pro", provider="modal", wait_for_ready=True)

    (tracked,) = weave.list_deployments()
    assert tracked.state == DeploymentState.FAILED
    record = await lifecycle_repo.get(tracked.id)
    assert record is not None
    assert record.state == DeploymentState.FAILED
    assert record.ready_at is None
    assert record.endpoint_url == ENDPOINT


@pytest.mark.asyncio
async def test_modal_disabled_healthcheck_does_not_fake_health():
    """With no readiness check to run, the deployment must stay STARTING even when waiting."""
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)
    _fast_healthcheck(weave, enabled=False)
    deploy_patch, url_patch = _patched_modal_deploy()

    with deploy_patch, url_patch:
        deployment = await weave.deploy(
            model="fish-s2-pro", provider="modal", wait_for_ready=True
        )

    assert probe_adapter.probe_call_count == 0
    assert deployment.state == DeploymentState.STARTING
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_modal_dry_run_is_distinguishable_from_live_healthy():
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, lifecycle_repo = _build_weave(probe_adapter)

    deployment = await weave.deploy(
        model="fish-s2-pro", provider="modal", dry_run=True, wait_for_ready=True
    )

    assert probe_adapter.probe_call_count == 0
    assert deployment.state == DeploymentState.PROVISIONING
    record = await lifecycle_repo.get(deployment.id)
    assert record is not None
    assert record.is_dry_run is True
    assert record.state == DeploymentState.PROVISIONING
