"""Unit tests for Provider adapters verifying autostop and provider custom arguments."""

import json
from unittest.mock import MagicMock, patch

import pytest

from inferweave.accounts import ProviderAccount
from inferweave.domain.deployment_record import DeploymentRecord
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
async def test_modal_provider_scaledown_window_is_decoupled_from_autostop_and_timeout(
    sample_profile, sample_runtime
):
    provider = ModalProvider()
    record = DeploymentRecord(
        id="iw-modal-test",
        model=sample_profile.id,
        provider="modal",
        resource=provider.resource_ref(
            "iw-modal-test", DeploymentRequest(model=sample_profile.id), ProviderAccount.ambient("modal")
        ),
    )
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
            await provider.provision(
                record, request, sample_profile, sample_runtime, ProviderAccount.ambient("modal")
            )

            mock_function.assert_called_once()
            call_kwargs = mock_function.call_args[1]

            # autostop_mins is InferWeave's full-destroy timer; it must NOT drive Modal's
            # scale-to-zero window (that is scaledown_window_seconds, default 1800s).
            assert call_kwargs.get("scaledown_window") == 1800
            # Verify timeout is the custom 7200, NOT the old hardcoded autostop timer
            assert call_kwargs.get("timeout") == 7200
            # Verify cpu and memory
            assert call_kwargs.get("cpu") == 4.0
            assert call_kwargs.get("memory") == 16384


class _RecordingRunner:
    """Fake SkyPilot WorkerRunner capturing the JSON request the provider would send."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    async def run(self, operation, payload, *, env, secrets=None, timeout, deployment_id=None, account_id=""):
        self.requests.append((operation, json.loads(json.dumps(payload))))
        return {"endpoint": "http://1.2.3.4:8000"}


@pytest.mark.asyncio
async def test_skypilot_provider_passes_autostop_autodown_and_resources(
    sample_profile, sample_runtime, tmp_path
):
    runner = _RecordingRunner()
    provider = SkyPilotProvider(cloud_name="runpod", state_dir=tmp_path, runner=runner)  # type: ignore[arg-type]
    account = ProviderAccount.ambient("runpod")
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
    record = DeploymentRecord(
        id="iw-test",
        model=sample_profile.id,
        provider="runpod",
        resource=provider.resource_ref("iw-test", request, account),
    )

    with patch.object(SkyPilotProvider, "_ensure_supported_platform", return_value=None):
        await provider.provision(record, request, sample_profile, sample_runtime, account)

    ((operation, payload),) = runner.requests
    assert operation == "launch"

    # Resources sent to the worker (it builds sky.Resources from them)
    resources = payload["resources"]
    assert resources["use_spot"] is True
    assert resources["disk_size"] == 120
    assert resources["region"] == "us-central-1"

    # Launch arguments: idle minutes -> idle_minutes_to_autostop, autodown -> down
    assert payload["idle_minutes"] == 25
    assert payload["down"] is True
