"""Unit tests for SkyPilotProvider lifecycle operations."""

from pathlib import Path
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
        _, launch_kwargs = mock_sky.launch.call_args
        assert launch_kwargs.get("idle_minutes_to_autostop") == 20
        assert launch_kwargs.get("down") is False

        mock_sky.endpoints.assert_called_once()
        assert deployment.provider == "runpod"
        assert deployment.endpoint_url == "http://1.2.3.4:8000"
        assert deployment.state == DeploymentState.HEALTHY

        # Validate docker image_id was passed to sky.Resources
        mock_sky.Resources.assert_called_once()
        _, res_kwargs = mock_sky.Resources.call_args
        assert res_kwargs.get("image_id") == "docker:vllm/vllm-openai:latest"


@pytest.mark.asyncio
async def test_skypilot_deploy_with_autostop_action_down(sample_profile, sample_runtime):
    from inferweave.domain.lifecycle import AutostopAction, AutostopPolicy
    from inferweave.domain.options import DeploymentOptions

    provider = SkyPilotProvider(cloud_name="runpod")
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        options=DeploymentOptions(
            autostop=AutostopPolicy(
                action=AutostopAction.DOWN,
                idle_minutes=15,
                enabled=True,
            )
        ),
    )

    mock_sky = MagicMock()
    mock_sky.launch = MagicMock(return_value=(1, None))
    mock_sky.endpoints = MagicMock(return_value={8000: "http://1.2.3.4:8000"})
    mock_sky.Task = MagicMock()
    mock_sky.Resources = MagicMock()
    mock_sky.stop = MagicMock()
    mock_sky.down = MagicMock()
    mock_sky.clouds.CLOUD_REGISTRY.from_str.return_value = MagicMock()

    with (
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        patch.object(provider, "_get_sky_module", return_value=mock_sky),
    ):
        deployment = await provider.deploy(request, sample_profile, sample_runtime)
        mock_sky.launch.assert_called_once()
        _, launch_kwargs = mock_sky.launch.call_args
        assert launch_kwargs.get("down") is True
        assert launch_kwargs.get("idle_minutes_to_autostop") == 15

        # Test calling deployment.stop() directly defaults to STOP
        await deployment.stop()
        mock_sky.stop.assert_called_once_with(cluster_name=deployment.id)

        # Test calling deployment.stop(action=AutostopAction.DOWN)
        await deployment.stop(action=AutostopAction.DOWN)
        mock_sky.down.assert_called_once_with(cluster_name=deployment.id)


@pytest.mark.asyncio
async def test_skypilot_deploy_with_autostop_disabled(sample_profile, sample_runtime):
    from inferweave.domain.lifecycle import AutostopPolicy
    from inferweave.domain.options import DeploymentOptions

    provider = SkyPilotProvider(cloud_name="runpod")
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        options=DeploymentOptions(
            autostop=AutostopPolicy(
                enabled=False,
                idle_minutes=None,
            )
        ),
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
        await provider.deploy(request, sample_profile, sample_runtime)
        mock_sky.launch.assert_called_once()
        _, launch_kwargs = mock_sky.launch.call_args
        assert launch_kwargs.get("idle_minutes_to_autostop") is None
        assert launch_kwargs.get("down") is False


@pytest.mark.asyncio
async def test_skypilot_docker_image_and_workdir(sample_profile, sample_runtime):
    provider = SkyPilotProvider(cloud_name="runpod")
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        custom_args={
            "provider_args": {
                "workdir": "/path/to/workdir",
            }
        },
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
        await provider.deploy(request, sample_profile, sample_runtime)
        mock_sky.Task.assert_called_once()
        _, task_kwargs = mock_sky.Task.call_args
        assert task_kwargs.get("workdir") == "/path/to/workdir"

        mock_sky.Resources.assert_called_once()
        _, res_kwargs = mock_sky.Resources.call_args
        assert res_kwargs.get("image_id") == "docker:vllm/vllm-openai:latest"


@pytest.mark.asyncio
async def test_skypilot_auto_workdir_for_inferweave_workers(sample_profile):
    from pathlib import Path

    flux_runtime = RuntimeSpec(
        name="flux-diffusers",
        docker_image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
        run_command="python3 -m inferweave.workers.flux --model black-forest-labs/FLUX.1-schnell --port 8000",
        port=8000,
        env_vars={"MODEL": "black-forest-labs/FLUX.1-schnell"},
    )
    provider = SkyPilotProvider(cloud_name="runpod")
    request = DeploymentRequest(
        model="black-forest-labs/FLUX.1-schnell",
        provider="runpod",
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
        await provider.deploy(request, sample_profile, flux_runtime)
        mock_sky.Task.assert_called_once()
        _, task_kwargs = mock_sky.Task.call_args

        # Ensure workdir was auto-resolved to local directory containing inferweave
        workdir = task_kwargs.get("workdir")
        assert workdir is not None
        assert (Path(workdir) / "inferweave").is_dir()

        # Ensure PYTHONPATH contains .:$PYTHONPATH
        task_envs = task_kwargs.get("envs", {})
        assert "PYTHONPATH" in task_envs
        assert ".:$PYTHONPATH" in task_envs["PYTHONPATH"]


def test_resolve_worker_workdir_from_source_tree():
    from inferweave.providers.skypilot import resolve_worker_workdir

    workdir = resolve_worker_workdir()
    assert workdir is not None
    p = Path(workdir)
    assert p.name == "src"
    assert (p / "inferweave").is_dir()


def test_resolve_worker_workdir_from_site_packages_never_syncs_site_packages(tmp_path):
    from inferweave.providers.skypilot import resolve_worker_workdir

    # Simulate installed environment in site-packages
    site_packages = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    pkg_dir = site_packages / "inferweave"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("# inferweave init", encoding="utf-8")
    (pkg_dir / "workers").mkdir()
    (pkg_dir / "workers" / "flux.py").write_text("# flux worker", encoding="utf-8")

    # Add huge dummy packages alongside inferweave
    torch_dir = site_packages / "torch"
    torch_dir.mkdir()
    (torch_dir / "libtorch.so").write_bytes(b"huge_binary")

    staging_base = tmp_path / "custom_staging"

    # Resolve workdir for the site-packages installed inferweave
    staged_workdir = resolve_worker_workdir(pkg_path=pkg_dir, staging_base=staging_base)

    assert staged_workdir is not None
    # 1. CRITICAL: Never return the entire site-packages!
    assert staged_workdir != str(site_packages)
    assert Path(staged_workdir).name != "site-packages"

    # 2. Returned workdir must be the staged worker_pkg directory
    staged_path = Path(staged_workdir)
    assert staged_path == staging_base / "worker_pkg"

    # 3. Only inferweave was staged, NOT torch or other dependencies
    assert (staged_path / "inferweave").is_dir()
    assert (staged_path / "inferweave" / "__init__.py").exists()
    assert (staged_path / "inferweave" / "workers" / "flux.py").exists()
    assert not (staged_path / "torch").exists()





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
    from inferweave.domain.lifecycle import AutostopAction

    provider = SkyPilotProvider(cloud_name="runpod")
    deployment_id = "iw-test-cluster"

    mock_sky = MagicMock()
    mock_sky.stop = MagicMock()
    mock_sky.down = MagicMock()

    with (
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        patch.object(provider, "_get_sky_module", return_value=mock_sky),
    ):
        # Default action is STOP
        await provider.stop(deployment_id)
        mock_sky.stop.assert_called_once_with(cluster_name=deployment_id)

        # Explicit action DOWN
        await provider.stop(deployment_id, action=AutostopAction.DOWN)
        mock_sky.down.assert_called_once_with(cluster_name=deployment_id)

        # String action "down"
        await provider.stop(deployment_id, action="down")
        assert mock_sky.down.call_count == 2


@pytest.mark.asyncio
async def test_skypilot_stop_dry_run(sample_profile, sample_runtime):
    provider = SkyPilotProvider(cloud_name="aws")
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="aws",
        dry_run=True,
    )

    deployment = await provider.deploy(request, sample_profile, sample_runtime)
    # Stopping dry-run deployment must succeed on any OS without invoking sky.stop/down
    await provider.stop(deployment.id)
    status = await provider.get_status(deployment.id)
    assert status.state == DeploymentState.STOPPED
    assert status.model == sample_profile.id


@pytest.mark.asyncio
async def test_skypilot_get_status_up(sample_profile, sample_runtime):
    provider = SkyPilotProvider(cloud_name="runpod")
    request = DeploymentRequest(
        model=sample_profile.id,
        provider="runpod",
        dry_run=True,
    )
    deployment = await provider.deploy(request, sample_profile, sample_runtime)

    status = await provider.get_status(deployment.id)
    assert status.model == sample_profile.id
    assert status.provider == "runpod"


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
