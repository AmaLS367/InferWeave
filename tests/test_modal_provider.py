"""Unit tests for ModalProvider lifecycle methods."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.providers.modal_provider import ModalProvider
from inferweave.runtimes.base import RuntimeSpec


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
            assert deployment.state == DeploymentState.HEALTHY


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


@pytest.mark.asyncio
async def test_modal_provider_stop():
    provider = ModalProvider()
    deployment_id = "iw-modal-test-123456"

    mock_client = MagicMock()
    mock_client.stub.AppStop = AsyncMock()
    mock_lifecycle = MagicMock()
    mock_lifecycle.app_state = 1  # Not stopped

    with (
        patch("modal.client._Client.from_env", AsyncMock(return_value=mock_client)),
        patch(
            "modal.cli.app.resolve_app_identifier",
            AsyncMock(return_value=("ap-12345", "main", mock_lifecycle)),
        ),
    ):
        await provider.stop(deployment_id)
        mock_client.stub.AppStop.assert_called_once()


@pytest.mark.asyncio
async def test_modal_provider_get_status():
    provider = ModalProvider()
    deployment_id = "iw-modal-test-123456"

    mock_client = MagicMock()
    mock_lifecycle = MagicMock()
    from modal_proto import api_pb2

    mock_lifecycle.app_state = api_pb2.APP_STATE_DEPLOYED

    with (
        patch("modal.client._Client.from_env", AsyncMock(return_value=mock_client)),
        patch(
            "modal.cli.app.resolve_app_identifier",
            AsyncMock(return_value=("ap-12345", "main", mock_lifecycle)),
        ),
    ):
        status = await provider.get_status(deployment_id)
        assert status.state == DeploymentState.HEALTHY
        assert status.id == deployment_id
