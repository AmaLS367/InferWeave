"""Regression tests: SkyPilot deployments are only HEALTHY after a verified readiness probe.

Mirrors tests/test_modal_readiness.py. A launched / UP cluster proves infrastructure only.
The SkyPilot worker is replaced by a fake ``WorkerRunner``; the SDK, provisioning service and
lifecycle service are real.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from inferweave import DeploymentState, InferWeave
from inferweave.accounts import (
    AccountManager,
    AccountsConfig,
    ProviderAccount,
    ProviderAccounts,
    runpod_account,
)
from inferweave.adapters.healthcheck.mock_probe import MockHealthcheckProbeAdapter
from inferweave.adapters.lifecycle.memory_repository import InMemoryDeploymentRepository
from inferweave.adapters.lifecycle.watchdog import MockWatchdogAdapter
from inferweave.core.exceptions import HealthcheckTimeoutError, ProviderOperationError
from inferweave.core.failures import FailureKind
from inferweave.domain.deployment_record import DeploymentRecord
from inferweave.domain.healthcheck import ProbeOutcome, ProbeResult
from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.profile import HealthcheckConfig
from inferweave.providers.router import ProviderRouter
from inferweave.providers.skypilot import SkyPilotProvider
from inferweave.services.healthcheck_service import HealthcheckService
from inferweave.services.lifecycle_service import LifecycleService

ENDPOINT = "http://1.2.3.4:8000"
MODEL = "fish-s2-pro"


@dataclass
class Call:
    op: str
    payload: dict[str, Any]
    secrets: dict[str, str]
    account_id: str


class FakeRunner:
    """Stands in for the SkyPilot worker; ``cluster_status`` drives the status operation."""

    def __init__(self, cluster_status: str | None = "UP", endpoint: str | None = ENDPOINT) -> None:
        self.calls: list[Call] = []
        self.endpoint = endpoint
        self.cluster_status = cluster_status
        self.errors: dict[str, BaseException] = {}

    @property
    def ops(self) -> list[str]:
        return [call.op for call in self.calls]

    async def run(
        self,
        operation: str,
        payload: Any,
        *,
        env: Any,
        secrets: Any = None,
        timeout: float,
        deployment_id: str | None = None,
        account_id: str = "",
    ) -> Any:
        self.calls.append(
            Call(operation, copy.deepcopy(dict(payload)), dict(secrets or {}), account_id)
        )
        if operation in self.errors:
            raise self.errors[operation]
        if operation == "launch":
            return {"endpoint": self.endpoint}
        if operation == "status":
            if self.cluster_status is None:
                return {"exists": False, "status": None, "endpoint": None}
            return {"exists": True, "status": self.cluster_status, "endpoint": self.endpoint}
        return {"existed": True}


class _StateRecordingRepository(InMemoryDeploymentRepository):
    """In-memory repository that records every persisted state transition."""

    def __init__(self) -> None:
        super().__init__()
        self.saved_states: list[DeploymentState] = []

    async def save(self, record: DeploymentRecord) -> None:
        self.saved_states.append(record.state)
        await super().save(record)


@pytest.fixture(autouse=True)
def _skypilot_on_any_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralizes the native-Windows guard and any inherited platform credentials."""
    monkeypatch.setattr(SkyPilotProvider, "_ensure_supported_platform", lambda self: None)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)


def _build_weave(
    probe_adapter: MockHealthcheckProbeAdapter,
    runner: FakeRunner,
    tmp_path: Path,
    accounts: AccountManager | None = None,
) -> tuple[InferWeave, _StateRecordingRepository]:
    healthcheck_svc = HealthcheckService(probe_port=probe_adapter)
    repo = _StateRecordingRepository()
    accounts = accounts or AccountManager(AccountsConfig(state_dir=tmp_path))
    lifecycle_svc = LifecycleService(
        watchdog_port=MockWatchdogAdapter(),
        repository=repo,
        healthcheck_service=healthcheck_svc,
        accounts=accounts,
    )
    router = ProviderRouter()
    router.register(SkyPilotProvider("runpod", state_dir=tmp_path, runner=runner))  # type: ignore[arg-type]
    weave = InferWeave(
        router=router,
        healthcheck_service=healthcheck_svc,
        lifecycle_service=lifecycle_svc,
        accounts=accounts,
    )
    _fast_healthcheck(weave)  # no 10s initial readiness delay in unit tests
    return weave, repo


def _fast_healthcheck(weave: InferWeave, **overrides: object) -> None:
    profile = weave.registry.get(MODEL).model_copy(deep=True)
    profile.healthcheck = HealthcheckConfig(
        initial_delay_seconds=0,
        timeout_seconds=0.1,
        probe_interval_seconds=0.02,
        **overrides,
    )
    weave.register_model(profile)


def _hc(weave: InferWeave) -> HealthcheckConfig:
    return weave.registry.get(MODEL).healthcheck


async def _deploy(weave: InferWeave, **kwargs: object):
    return await weave.deploy(model=MODEL, provider="runpod", gpu_type="L4", **kwargs)


@pytest.mark.asyncio
async def test_skypilot_provider_never_persists_healthy_on_deploy(tmp_path):
    """Provisioning alone transitions PROVISIONING -> STARTING, never HEALTHY."""
    runner = FakeRunner()
    weave, repo = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True), runner, tmp_path)

    deployment = await _deploy(weave, wait_for_ready=False)

    assert repo.saved_states[0] == DeploymentState.PROVISIONING  # write-ahead record
    assert DeploymentState.STARTING in repo.saved_states
    assert DeploymentState.HEALTHY not in repo.saved_states
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.STARTING
    assert record.endpoint_url == ENDPOINT
    assert record.ready_at is None
    assert record.account == "ambient"
    assert record.resource is not None and record.resource.name == deployment.id
    assert record.needs_reconciliation is False
    assert runner.ops == ["launch"]


@pytest.mark.asyncio
async def test_skypilot_no_endpoint_stays_provisioning(tmp_path):
    runner = FakeRunner(endpoint=None)
    weave, repo = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True), runner, tmp_path)

    deployment = await _deploy(weave, wait_for_ready=False)

    assert deployment.state == DeploymentState.PROVISIONING
    record = await repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.PROVISIONING


@pytest.mark.asyncio
async def test_skypilot_no_wait_has_endpoint_but_is_not_healthy(tmp_path):
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, repo = _build_weave(probe_adapter, FakeRunner(), tmp_path)

    deployment = await _deploy(weave, wait_for_ready=False)

    assert probe_adapter.probe_call_count == 0
    assert deployment.endpoint_url == ENDPOINT
    assert deployment.state == DeploymentState.STARTING
    assert deployment.is_healthy is False
    assert deployment.status.ready_at is None

    # Persisted state agrees with the in-memory deployment.
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.state == deployment.state == DeploymentState.STARTING
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_skypilot_manual_readiness_transitions_memory_and_record_to_healthy(tmp_path):
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, repo = _build_weave(probe_adapter, FakeRunner(), tmp_path)

    deployment = await _deploy(weave, wait_for_ready=False)
    status = await deployment.wait_for_ready()

    assert probe_adapter.probe_call_count >= 1
    assert status.state == deployment.state == DeploymentState.HEALTHY
    assert deployment.status.ready_at is not None
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.HEALTHY
    assert record.ready_at is not None


@pytest.mark.asyncio
async def test_skypilot_wait_for_ready_returns_healthy_after_successful_probe(tmp_path):
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, repo = _build_weave(probe_adapter, FakeRunner(), tmp_path)

    deployment = await _deploy(weave, wait_for_ready=True)

    assert probe_adapter.probe_call_count >= 1
    assert deployment.state == DeploymentState.HEALTHY
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.HEALTHY
    assert record.ready_at is not None


@pytest.mark.asyncio
async def test_skypilot_readiness_timeout_marks_memory_and_record_failed(tmp_path):
    failing = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="container crashed",
    )
    probe_adapter = MockHealthcheckProbeAdapter(canned_results=[failing] * 50)
    weave, repo = _build_weave(probe_adapter, FakeRunner(), tmp_path)

    with pytest.raises(HealthcheckTimeoutError):
        await _deploy(weave, wait_for_ready=True)

    (tracked,) = weave.list_deployments()
    assert tracked.state == DeploymentState.FAILED
    record = await repo.get(tracked.id)
    assert record is not None
    assert record.state == DeploymentState.FAILED
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_skypilot_disabled_healthcheck_does_not_fake_health(tmp_path):
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, repo = _build_weave(probe_adapter, FakeRunner(), tmp_path)
    _fast_healthcheck(weave, enabled=False)

    deployment = await _deploy(weave, wait_for_ready=True)

    assert probe_adapter.probe_call_count == 0
    assert deployment.state == DeploymentState.STARTING
    record = await repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_skypilot_dry_run_is_distinguishable_from_live_healthy(tmp_path):
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    runner = FakeRunner()
    weave, repo = _build_weave(probe_adapter, runner, tmp_path)

    deployment = await _deploy(weave, dry_run=True, wait_for_ready=True)

    assert probe_adapter.probe_call_count == 0
    assert runner.calls == []  # a dry run never reaches the worker
    assert deployment.state == DeploymentState.PROVISIONING
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.is_dry_run is True
    assert record.account is None
    assert record.state == DeploymentState.PROVISIONING


@pytest.mark.asyncio
async def test_skypilot_failed_launch_is_not_swallowed_and_record_is_failed(tmp_path):
    runner = FakeRunner(cluster_status=None)  # reconciliation finds nothing left behind
    runner.errors["launch"] = ProviderOperationError(
        "no capacity", FailureKind.CAPACITY, resource_may_exist=True
    )
    weave, repo = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True), runner, tmp_path)

    with pytest.raises(ProviderOperationError, match="no capacity"):
        await _deploy(weave, wait_for_ready=False)

    assert runner.ops == ["launch", "status"]  # the cluster name was looked up under the account
    (record,) = await repo.list_all()
    assert record.state == DeploymentState.FAILED
    assert record.needs_reconciliation is False
    assert weave.list_deployments() == []


# --- Provider status: infrastructure state only ------------------------------------------


def _record() -> DeploymentRecord:
    return DeploymentRecord(
        id="iw-test-cluster", model=MODEL, provider="runpod", account="ambient"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cluster_status", "expected"),
    [
        ("UP", DeploymentState.STARTING),
        ("CLUSTERSTATUS.UP", DeploymentState.STARTING),
        ("INIT", DeploymentState.PROVISIONING),
        ("STOPPED", DeploymentState.STOPPED),
    ],
)
async def test_skypilot_status_reports_infrastructure_state_never_healthy(
    tmp_path, cluster_status, expected
):
    provider = SkyPilotProvider("runpod", state_dir=tmp_path, runner=FakeRunner(cluster_status))  # type: ignore[arg-type]

    status = await provider.status(_record(), ProviderAccount.ambient("runpod"))

    assert status.state == expected
    assert status.state != DeploymentState.HEALTHY


@pytest.mark.asyncio
async def test_skypilot_status_up_cluster_exposes_endpoint(tmp_path):
    provider = SkyPilotProvider("runpod", state_dir=tmp_path, runner=FakeRunner())  # type: ignore[arg-type]

    status = await provider.status(_record(), ProviderAccount.ambient("runpod"))

    assert status.endpoint_url == ENDPOINT


@pytest.mark.asyncio
async def test_skypilot_status_missing_cluster_is_stopped(tmp_path):
    provider = SkyPilotProvider("runpod", state_dir=tmp_path, runner=FakeRunner(None))  # type: ignore[arg-type]

    status = await provider.status(_record(), ProviderAccount.ambient("runpod"))

    assert status.state == DeploymentState.STOPPED


# --- refresh_status: probe=False cannot create health, probe=True can ---------------------


@pytest.mark.asyncio
async def test_refresh_status_without_probe_cannot_create_false_health(tmp_path):
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, repo = _build_weave(probe_adapter, FakeRunner(), tmp_path)

    deployment = await _deploy(weave, wait_for_ready=False)
    status = await weave.lifecycle_service.refresh_status(deployment.id, probe=False)

    assert probe_adapter.probe_call_count == 0
    assert status.state == DeploymentState.STARTING
    assert deployment.state == DeploymentState.STARTING
    assert deployment.is_healthy is False
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.STARTING
    assert record.ready_at is None


@pytest.mark.asyncio
async def test_refresh_status_with_probe_promotes_to_healthy_after_real_probe(tmp_path):
    probe_adapter = MockHealthcheckProbeAdapter(default_healthy=True)
    weave, repo = _build_weave(probe_adapter, FakeRunner(), tmp_path)

    deployment = await _deploy(weave, wait_for_ready=False)
    assert probe_adapter.probe_call_count == 0
    status = await weave.lifecycle_service.refresh_status(
        deployment.id, probe=True, healthcheck_config=_hc(weave)
    )

    assert probe_adapter.probe_call_count == 1
    assert status.state == DeploymentState.HEALTHY
    assert deployment.state == DeploymentState.HEALTHY
    assert status.ready_at is not None
    record = await repo.get(deployment.id)
    assert record is not None
    assert record.state == DeploymentState.HEALTHY
    assert record.ready_at is not None


@pytest.mark.asyncio
async def test_refresh_status_with_failing_probe_stays_starting(tmp_path):
    not_ready = ProbeResult(
        is_healthy=False,
        outcome=ProbeOutcome.CONNECTION_REFUSED,
        error_message="weights still loading",
    )
    probe_adapter = MockHealthcheckProbeAdapter(canned_results=[not_ready])
    weave, repo = _build_weave(probe_adapter, FakeRunner(), tmp_path)

    deployment = await _deploy(weave, wait_for_ready=False)
    status = await weave.lifecycle_service.refresh_status(
        deployment.id, probe=True, healthcheck_config=_hc(weave)
    )

    assert probe_adapter.probe_call_count == 1
    assert status.state == DeploymentState.STARTING
    record = await repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_refresh_status_without_probe_keeps_earlier_verified_health(tmp_path):
    """A state proven by a real probe is not downgraded by a probe-less infra refresh."""
    weave, repo = _build_weave(
        MockHealthcheckProbeAdapter(default_healthy=True), FakeRunner(), tmp_path
    )

    deployment = await _deploy(weave, wait_for_ready=True)
    status = await weave.lifecycle_service.refresh_status(deployment.id, probe=False)

    assert status.state == DeploymentState.HEALTHY
    record = await repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.HEALTHY


@pytest.mark.asyncio
async def test_refresh_status_propagates_stopped_infrastructure(tmp_path):
    runner = FakeRunner()
    weave, repo = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True), runner, tmp_path)

    deployment = await _deploy(weave, wait_for_ready=True)
    assert deployment.state == DeploymentState.HEALTHY

    runner.cluster_status = "STOPPED"
    status = await weave.lifecycle_service.refresh_status(
        deployment.id, probe=True, healthcheck_config=_hc(weave)
    )

    assert status.state == DeploymentState.STOPPED
    assert deployment.state == DeploymentState.STOPPED
    record = await repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_refresh_status_keeps_last_state_on_transient_control_plane_error(tmp_path):
    runner = FakeRunner()
    weave, repo = _build_weave(MockHealthcheckProbeAdapter(default_healthy=True), runner, tmp_path)
    deployment = await _deploy(weave, wait_for_ready=True)

    runner.errors["status"] = ProviderOperationError("flaky", FailureKind.TRANSIENT)
    status = await weave.lifecycle_service.refresh_status(deployment.id, probe=False)

    assert status.state == DeploymentState.HEALTHY
    record = await repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.HEALTHY


# --- pooled accounts through the SDK ---------------------------------------------------------


def _pooled_accounts(tmp_path: Path) -> AccountManager:
    config = AccountsConfig(
        {
            "runpod": ProviderAccounts(
                accounts=[
                    runpod_account("a", api_key_env="IW_TEST_RUNPOD_A"),
                    runpod_account("b", api_key_env="IW_TEST_RUNPOD_B"),
                ]
            )
        },
        state_dir=tmp_path,
    )
    return AccountManager(
        config,
        environ={
            "IW_TEST_RUNPOD_A": "SENTINEL-runpod-a-secret",
            "IW_TEST_RUNPOD_B": "SENTINEL-runpod-b-secret",
        },
    )


@pytest.mark.asyncio
async def test_pooled_deployment_is_bound_to_its_account_for_status_and_down(tmp_path):
    runner = FakeRunner()
    weave, repo = _build_weave(
        MockHealthcheckProbeAdapter(default_healthy=True),
        runner,
        tmp_path,
        accounts=_pooled_accounts(tmp_path),
    )

    deployment = await _deploy(weave, wait_for_ready=False, account="b")
    assert deployment.account == "b"
    record = await repo.get(deployment.id)
    assert record is not None and record.account == "b"

    await weave.lifecycle_service.refresh_status(deployment.id, probe=False)
    await weave.stop(deployment.id, action=AutostopAction.DOWN)

    assert runner.ops == ["launch", "status", "down"]
    assert {call.account_id for call in runner.calls} == {"b"}
    assert {call.secrets["api_key"] for call in runner.calls} == {"SENTINEL-runpod-b-secret"}
    assert all(call.payload["mode"] == "account" for call in runner.calls)
    stopped = await repo.get(deployment.id)
    assert stopped is not None and stopped.state == DeploymentState.STOPPED
