"""Unit tests for SkyPilotProvider lifecycle operations."""

from unittest.mock import MagicMock, patch

import pytest

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import DeploymentState, WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.providers.skypilot import SkyPilotProvider
from inferweave.runtimes.base import RuntimeSpec


@pytest.fixture
def sample_profile() -> ModelProfile:
    return ModelProfile(
        id="meta-llama/Meta-Llama-3-8B-Instruct",
        name="Llama 3 8B Instruct",
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
        setup_commands=["echo 'setup'"],
        run_command="python3 -m vllm.entrypoints.openai.api_server",
        port=8000,
        env_vars={"MODEL": "meta-llama/Meta-Llama-3-8B-Instruct"},
    )


@pytest.mark.asyncio
async def test_skypilot_deploy_success(sample_profile, sample_runtime):
    provider = SkyPilotProvider(cloud_name="runpod")
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        gpu_type="A100",
        num_gpus=1,
        autostop_mins=20,
    )

    mock_sky = MagicMock()
    mock_sky.launch = MagicMock(return_value=(1, None))
    mock_sky.endpoints = MagicMock(return_value={8000: "http://1.2.3.4:8000"})
    mock_sky.Task = MagicMock()
    mock_sky.Resources = MagicMock()
    mock_sky.clouds.CLOUD_REGISTRY.from_str.return_value = MagicMock()

    with (
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        patch.object(provider, "_get_sky_module", return_value=mock_sky),
    ):
        deployment = await provider.deploy(request, sample_profile, sample_runtime)

        mock_sky.launch.assert_called_once()
        mock_sky.endpoints.assert_called_once()
        assert deployment.provider == "runpod"
        assert deployment.endpoint_url == "http://1.2.3.4:8000"
        assert deployment.state == DeploymentState.HEALTHY


@pytest.mark.asyncio
async def test_skypilot_dry_run(sample_profile, sample_runtime):
    provider = SkyPilotProvider(cloud_name="aws")
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="aws",
        dry_run=True,
    )

    deployment = await provider.deploy(request, sample_profile, sample_runtime)
    assert deployment.provider == "aws"
    assert deployment.state == DeploymentState.PROVISIONING
    assert "dryrun" in (deployment.endpoint_url or "")


@pytest.mark.asyncio
async def test_skypilot_stop():
    provider = SkyPilotProvider(cloud_name="runpod")
    deployment_id = "iw-test-cluster"

    mock_sky = MagicMock()
    mock_sky.down = MagicMock()

    with (
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        patch.object(provider, "_get_sky_module", return_value=mock_sky),
    ):
        await provider.stop(deployment_id)
        mock_sky.down.assert_called_once_with(cluster_name=deployment_id)


@pytest.mark.asyncio
async def test_skypilot_get_status_up():
    provider = SkyPilotProvider(cloud_name="runpod")
    deployment_id = "iw-test-cluster"

    mock_sky = MagicMock()
    mock_record = MagicMock()
    mock_record.status = "UP"
    mock_sky.status = MagicMock(return_value=[mock_record])
    mock_sky.endpoints = MagicMock(return_value={8000: "http://5.6.7.8:8000"})

    with (
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        patch.object(provider, "_get_sky_module", return_value=mock_sky),
    ):
        status = await provider.get_status(deployment_id)
        assert status.state == DeploymentState.HEALTHY
        assert status.endpoint_url == "http://5.6.7.8:8000"


@pytest.mark.asyncio
async def test_skypilot_get_status_stopped():
    provider = SkyPilotProvider(cloud_name="runpod")
    deployment_id = "iw-test-cluster"

    mock_sky = MagicMock()
    mock_record = MagicMock()
    mock_record.status = "STOPPED"
    mock_sky.status = MagicMock(return_value=[mock_record])

    with (
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        patch.object(provider, "_get_sky_module", return_value=mock_sky),
    ):
        status = await provider.get_status(deployment_id)
        assert status.state == DeploymentState.STOPPED
