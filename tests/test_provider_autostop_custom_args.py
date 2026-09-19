"""Unit tests for Provider adapters verifying autostop and provider custom arguments."""

from unittest.mock import MagicMock, patch

import pytest

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.providers.modal_provider import ModalProvider
from inferweave.providers.skypilot import SkyPilotProvider
from inferweave.runtimes.base import RuntimeSpec


@pytest.fixture
def sample_profile():
    return ModelProfile(
        id="test-model",
        name="Test Model",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(min_vram_gb=16.0, gpu_count=1),
        healthcheck=HealthcheckConfig(port=8000),
    )



@pytest.fixture
def sample_runtime():
    return RuntimeSpec(
        name="vllm",
        docker_image="vllm/vllm-openai:latest",
        run_command="python3 -m vllm.entrypoints.openai.api_server",
        port=8000,
    )


@pytest.mark.asyncio
async def test_modal_provider_autostop_scaledown_window_and_timeout(sample_profile, sample_runtime):
    provider = ModalProvider()
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="modal",
        autostop_mins=15,
        custom_args={
            "timeout_seconds": 7200,
            "cpu": 4.0,
            "memory": 16384,
        },
    )

    with patch("modal.App.function") as mock_function:
        # Mock function decorator
        def decorator(*args, **kwargs):
            def inner(fn):
                fn.get_web_url = MagicMock(return_value="https://modal-test.modal.run")
                return fn

            return inner

        mock_function.side_effect = decorator
        with patch("modal.App.deploy", return_value=None):
            await provider.deploy(request, sample_profile, sample_runtime)

            mock_function.assert_called_once()
            call_kwargs = mock_function.call_args[1]

            # Verify scaledown_window is 15 * 60 = 900 seconds
            assert call_kwargs.get("scaledown_window") == 900
            # Verify timeout is the custom 7200, NOT the old hardcoded autostop timer
            assert call_kwargs.get("timeout") == 7200
            # Verify cpu and memory
            assert call_kwargs.get("cpu") == 4.0
            assert call_kwargs.get("memory") == 16384



@pytest.mark.asyncio
async def test_skypilot_provider_passes_autostop_autodown_and_resources(sample_profile, sample_runtime):
    provider = SkyPilotProvider(cloud_name="runpod")
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        autostop_mins=25,
        custom_args={
            "allow_spot": True,
            "autodown": True,
            "disk_size_gb": 120,
            "preferred_regions": ["us-central-1"],
        },
    )

    mock_sky = MagicMock()
    mock_sky.launch = MagicMock(return_value=(1, None))
    mock_sky.endpoints = MagicMock(return_value={8000: "http://1.2.3.4:8000"})
    mock_sky.Task = MagicMock()
    mock_sky.Resources = MagicMock()

    with (
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        patch.object(provider, "_get_sky_module", return_value=mock_sky),
    ):
        await provider.deploy(request, sample_profile, sample_runtime)

        # Verify Resources kwargs
        res_kwargs = mock_sky.Resources.call_args[1]
        assert res_kwargs.get("use_spot") is True
        assert res_kwargs.get("disk_size") == 120
        assert res_kwargs.get("region") == "us-central-1"

        # Verify launch arguments
        launch_kwargs = mock_sky.launch.call_args[1]
        assert launch_kwargs.get("idle_minutes_to_autostop") == 25
        assert launch_kwargs.get("down") is True
