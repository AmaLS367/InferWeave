"""Unit tests for ModalProvider lifecycle methods with mocked Modal SDK.

These tests validate the adapter's translation, image building, and app registration
against a mocked Modal SDK, without executing live deployments on Modal cloud.
"""

import inspect
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from packaging.requirements import Requirement

from inferweave.adapters.lifecycle.memory_repository import InMemoryDeploymentRepository
from inferweave.domain.lifecycle import AutostopAction
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.providers import modal_provider as modal_provider_module
from inferweave.providers.modal_provider import ModalProvider
from inferweave.runtimes.base import RuntimeSpec

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture
def sample_profile() -> ModelProfile:
    return ModelProfile(
        id="test/sample-model",
        name="Sample Model",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(
            min_vram_gb=16,
            gpu_count=1,
            recommended_gpus=["A10G"],
        ),
        healthcheck=HealthcheckConfig(port=8000),
    )


@pytest.fixture
def sample_runtime() -> RuntimeSpec:
    return RuntimeSpec(
        name="vllm",
        docker_image="vllm/vllm-openai:latest",
        setup_commands=["echo 'setting up'"],
        run_command="python -m vllm --port 8000",
        port=8000,
        env_vars={"MODEL": "test/sample-model"},
    )


@pytest.mark.asyncio
async def test_modal_provider_deploy_success(sample_profile, sample_runtime):
    """Verifies that ModalProvider sets up the app and transitions to STARTING."""
    provider = ModalProvider()
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="modal",
        gpu_type="A100",
        num_gpus=1,
        autostop_mins=15,
    )

    with patch("modal.App.deploy") as mock_deploy:
        mock_deploy.return_value = None
        with patch(
            "modal.Function.get_web_url",
            return_value="https://workspace--iw-modal-test.modal.run",
        ):
            deployment = await provider.deploy(request, sample_profile, sample_runtime)

            assert mock_deploy.called
            assert deployment.id.startswith("iw-modal-")
            assert (
                deployment.endpoint_url == "https://workspace--iw-modal-test.modal.run"
            )
            # app.deploy() succeeding does not prove runtime readiness
            assert deployment.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_modal_provider_dry_run(sample_profile, sample_runtime):
    provider = ModalProvider()
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="modal",
        dry_run=True,
    )

    with patch("modal.App.deploy") as mock_deploy:
        deployment = await provider.deploy(request, sample_profile, sample_runtime)
        mock_deploy.assert_not_called()
        assert deployment.state == DeploymentState.PROVISIONING
        assert "dryrun" in (deployment.endpoint_url or "")


async def _deploy_live(provider, sample_profile, sample_runtime):
    request = DeploymentRequest(model=sample_profile.id, provider="modal")
    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://live.modal.run"),
    ):
        return await provider.deploy(request, sample_profile, sample_runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [None, AutostopAction.STOP, AutostopAction.DOWN])
async def test_modal_provider_stop_uses_public_stop_app(
    action, sample_profile, sample_runtime
):
    """STOP and DOWN both map to Modal's single app-level stop (public SDK API)."""
    provider = ModalProvider()
    deployment = await _deploy_live(provider, sample_profile, sample_runtime)

    with patch("modal.experimental.stop_app") as mock_stop:
        await provider.stop(deployment.id, action=action)

    mock_stop.assert_called_once_with(deployment.id)
    record = provider._local_deployments[deployment.id]["record"]
    assert record.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_modal_provider_stop_works_from_persisted_record_only(
    sample_profile, sample_runtime
):
    """Cross-process stop: a fresh provider with no in-memory state stops by deployment ID."""
    repo = InMemoryDeploymentRepository()
    first = ModalProvider(repository=repo)
    deployment = await _deploy_live(first, sample_profile, sample_runtime)

    fresh = ModalProvider(repository=repo)
    assert deployment.id not in fresh._local_deployments
    with patch("modal.experimental.stop_app") as mock_stop:
        await fresh.stop(deployment.id)

    mock_stop.assert_called_once_with(deployment.id)
    record = await repo.get(deployment.id)
    assert record is not None and record.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_modal_provider_stop_is_idempotent_when_app_already_gone(
    sample_profile, sample_runtime
):
    import modal

    provider = ModalProvider()
    deployment = await _deploy_live(provider, sample_profile, sample_runtime)

    with patch(
        "modal.experimental.stop_app", side_effect=modal.exception.NotFoundError("gone")
    ):
        await provider.stop(deployment.id)

    record = provider._local_deployments[deployment.id]["record"]
    assert record.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_modal_provider_get_status_deployed_app_is_starting_not_healthy():
    """A deployed Modal app is infrastructure state only; it must never read as HEALTHY."""
    provider = ModalProvider()
    with patch("modal.App.lookup", return_value=MagicMock()) as mock_lookup:
        status = await provider.get_status("iw-modal-test-123456")

    mock_lookup.assert_called_once_with("iw-modal-test-123456")
    assert status.state == DeploymentState.STARTING


@pytest.mark.asyncio
async def test_modal_provider_get_status_missing_app_is_stopped():
    import modal

    provider = ModalProvider()
    with patch("modal.App.lookup", side_effect=modal.exception.NotFoundError("gone")):
        status = await provider.get_status("iw-modal-test-123456")

    assert status.state == DeploymentState.STOPPED


@pytest.mark.asyncio
async def test_modal_provider_get_status_transient_error_keeps_cached_state(
    sample_profile, sample_runtime
):
    provider = ModalProvider()
    deployment = await _deploy_live(provider, sample_profile, sample_runtime)

    with patch("modal.App.lookup", side_effect=OSError("network down")):
        status = await provider.get_status(deployment.id)

    assert status.state == DeploymentState.STARTING


def test_modal_provider_uses_only_public_modal_apis():
    """Private Modal internals (modal.cli, modal.client._Client, modal_proto) must not be used."""
    source = inspect.getsource(modal_provider_module)
    for private in ("modal_proto", "modal.cli", "_Client", "resolve_app_identifier", "api_pb2"):
        assert private not in source, f"modal_provider relies on private API: {private}"


def test_modal_public_surface_required_by_provider_exists():
    """Guards the installed Modal SDK still exposes the public APIs the provider calls."""
    import modal

    assert callable(modal.experimental.stop_app)
    assert callable(modal.App.lookup)
    assert issubclass(modal.exception.NotFoundError, Exception)


def test_modal_extra_supports_current_stable_sdk():
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    (spec,) = [
        Requirement(dep)
        for dep in project["optional-dependencies"]["modal"]
        if dep.lower().startswith("modal")
    ]
    assert spec.specifier.contains("1.6.0"), "current stable Modal must be allowed"
    assert spec.specifier.contains("1.6.5"), "patch updates in verified 1.6 series allowed"
    assert not spec.specifier.contains("1.5.9"), "unverified pre-1.6 Modal must stay excluded"
    assert not spec.specifier.contains("1.7.0"), "untested 1.7+ Modal must stay excluded"
    assert not spec.specifier.contains("2.0.0"), "untested major version 2.0 must stay excluded"


@pytest.mark.asyncio
async def test_modal_stop_capability_detection():
    """Validates capability detection fallback across public stop interfaces."""
    provider = ModalProvider()

    # 1. Standard modal.experimental.stop_app
    mock_modal = MagicMock()
    mock_modal.experimental.stop_app = MagicMock()
    mock_modal.exception.NotFoundError = type("NotFoundError", (Exception,), {})
    with patch.object(provider, "_get_modal_module", return_value=mock_modal):
        await provider._stop_modal_app("app-123")
        mock_modal.experimental.stop_app.assert_called_once_with("app-123")

    # 2. Future modal.stop_app (if promoted out of experimental)
    mock_modal2 = MagicMock(spec=["stop_app", "exception"])
    mock_modal2.stop_app = MagicMock()
    mock_modal2.exception.NotFoundError = type("NotFoundError", (Exception,), {})
    with patch.object(provider, "_get_modal_module", return_value=mock_modal2):
        await provider._stop_modal_app("app-456")
        mock_modal2.stop_app.assert_called_once_with("app-456")

    # 3. Missing stop API raises RuntimeError
    mock_modal3 = MagicMock(spec=["exception"])
    mock_modal3.exception.NotFoundError = type("NotFoundError", (Exception,), {})
    with (
        patch.object(provider, "_get_modal_module", return_value=mock_modal3),
        pytest.raises(RuntimeError, match="supported public app stop API"),
    ):
        await provider._stop_modal_app("app-789")


@pytest.mark.asyncio
async def test_modal_provider_stop_dry_run(sample_profile, sample_runtime):
    provider = ModalProvider()
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="modal",
        dry_run=True,
    )

    deployment = await provider.deploy(request, sample_profile, sample_runtime)
    await provider.stop(deployment.id)
    status = await provider.get_status(deployment.id)
    assert status.state == DeploymentState.STOPPED
    assert status.model == sample_profile.id


@pytest.mark.asyncio
async def test_modal_provider_get_status(sample_profile, sample_runtime):
    provider = ModalProvider()
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="modal",
        dry_run=True,
    )
    deployment = await provider.deploy(request, sample_profile, sample_runtime)

    status = await provider.get_status(deployment.id)
    assert status.state == DeploymentState.PROVISIONING
    assert status.id == deployment.id
    assert status.model == sample_profile.id
    assert status.provider == "modal"


@pytest.mark.asyncio
async def test_modal_provider_stop_failure_propagates_and_retains_state(
    sample_profile, sample_runtime
):
    """Ensures that Modal API failures propagate and do not falsely mark deployment as STOPPED."""
    provider = ModalProvider()
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="modal",
        dry_run=False,
    )

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://test-fail.modal.run"),
    ):
        deployment = await provider.deploy(request, sample_profile, sample_runtime)

    assert deployment.state == DeploymentState.STARTING

    with (
        patch.object(
            provider,
            "_stop_modal_app",
            side_effect=RuntimeError("Modal API authentication error"),
        ),
        pytest.raises(RuntimeError) as exc_info,
    ):
        await deployment.stop()

    assert "Modal API authentication error" in str(exc_info.value)

    # State must remain STARTING, NOT STOPPED!
    record = provider._local_deployments[deployment.id]["record"]
    assert record.state == DeploymentState.STARTING
    assert record.stopped_at is None
