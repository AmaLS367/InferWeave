"""Unit tests for InferWeave runtime workers and templates integration."""

import pytest
from httpx import ASGITransport, AsyncClient

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.runtimes.templates import FluxDiffusersTemplate, WanVideoTemplate
from inferweave.workers.base import WorkerArgs, parse_worker_args
from inferweave.workers.flux import FluxWorker, create_flux_app
from inferweave.workers.wan import WanWorker, create_wan_app


def test_parse_worker_args_flux():
    """Validates CLI argument parsing for FLUX worker with extra flags."""
    raw_args = [
        "--model",
        "black-forest-labs/FLUX.1-schnell",
        "--port",
        "8005",
        "--host",
        "127.0.0.1",
        "--quantize-4bit",
        "--dtype",
        "bfloat16",
        "--extra-flag",
        "some_value",
    ]
    parsed = parse_worker_args(raw_args, description="Test Flux")

    assert parsed.model == "black-forest-labs/FLUX.1-schnell"
    assert parsed.port == 8005
    assert parsed.host == "127.0.0.1"
    assert parsed.quantize_4bit is True
    assert parsed.quantize_8bit is False
    assert parsed.dtype == "bfloat16"
    assert parsed.extra_args == ["--extra-flag", "some_value"]


def test_parse_worker_args_wan():
    """Validates CLI argument parsing for WAN video worker."""
    raw_args = [
        "--model",
        "wan-video/wan-2.1",
        "--port",
        "8080",
        "--quantize-8bit",
        "--mock",
    ]
    parsed = parse_worker_args(raw_args, description="Test WAN")

    assert parsed.model == "wan-video/wan-2.1"
    assert parsed.port == 8080
    assert parsed.quantize_8bit is True
    assert parsed.mock is True


@pytest.mark.asyncio
async def test_flux_worker_endpoints():
    """Tests FLUX worker health and generation endpoints."""
    args = WorkerArgs(
        model="black-forest-labs/FLUX.1-schnell",
        mock=True,
    )
    worker = FluxWorker(args)
    app = create_flux_app(worker)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Healthcheck
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["model"] == "black-forest-labs/FLUX.1-schnell"

        healthz_resp = await client.get("/healthz")
        assert healthz_resp.status_code == 200
        assert healthz_resp.json() == {"status": "ok"}

        # 2. Direct Generate
        gen_resp = await client.post(
            "/generate",
            json={
                "prompt": "A futuristic city in the rain",
                "width": 512,
                "height": 512,
            },
        )
        assert gen_resp.status_code == 200
        gen_data = gen_resp.json()
        assert "image" in gen_data
        assert gen_data["format"] == "png"

        # 3. OpenAI Image Generation
        openai_resp = await client.post(
            "/v1/images/generations",
            json={"prompt": "A cute cat", "n": 2, "size": "1024x1024"},
        )
        assert openai_resp.status_code == 200
        openai_data = openai_resp.json()
        assert len(openai_data["data"]) == 2
        assert "b64_json" in openai_data["data"][0]


@pytest.mark.asyncio
async def test_wan_worker_endpoints():
    """Tests WAN worker health and video generation endpoints."""
    args = WorkerArgs(
        model="wan-video/wan-2.1",
        mock=True,
    )
    worker = WanWorker(args)
    app = create_wan_app(worker)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Healthcheck
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["model"] == "wan-video/wan-2.1"

        # 2. Direct Video Generate
        gen_resp = await client.post(
            "/generate",
            json={"prompt": "A drone shot over the ocean", "num_frames": 16},
        )
        assert gen_resp.status_code == 200
        gen_data = gen_resp.json()
        assert "video" in gen_data
        assert gen_data["format"] == "mp4"

        # 3. OpenAI-style Video Generation
        openai_resp = await client.post(
            "/v1/videos/generations",
            json={"prompt": "Time lapse clouds", "size": "832x480"},
        )
        assert openai_resp.status_code == 200
        openai_data = openai_resp.json()
        assert len(openai_data["data"]) == 1
        assert "b64_json" in openai_data["data"][0]


def test_runtime_templates_render_commands():
    """Ensures templates produce executable commands referencing the correct worker modules."""
    flux_template = FluxDiffusersTemplate()
    wan_template = WanVideoTemplate()

    flux_profile = ModelProfile(
        id="black-forest-labs/FLUX.1-schnell",
        name="FLUX.1 Schnell",
        workload_type=WorkloadType.IMAGE,
        default_runtime="flux-diffusers",
        hardware=HardwareRequirements(min_vram_gb=24.0),
        healthcheck=HealthcheckConfig(port=8000),
    )

    wan_profile = ModelProfile(
        id="wan-video/wan-2.1",
        name="WAN 2.1 Video",
        workload_type=WorkloadType.VIDEO,
        default_runtime="wan-video",
        hardware=HardwareRequirements(min_vram_gb=24.0),
        healthcheck=HealthcheckConfig(port=8000),
    )

    req = DeploymentRequest(
        model="test", custom_args={"extra_cli_args": ["--quantize-4bit"]}
    )

    flux_spec = flux_template.render(flux_profile, req)
    wan_spec = wan_template.render(wan_profile, req)

    assert (
        "python3 -m inferweave.workers.flux --model black-forest-labs/FLUX.1-schnell --port 8000"
        in flux_spec.run_command
    )
    assert "--quantize-4bit" in flux_spec.run_command

    # By default, without explicit artifact_id, target_artifact uses profile.id
    assert (
        "python3 -m inferweave.workers.wan --model wan-video/wan-2.1 --port 8000"
        in wan_spec.run_command
    )

    wan_profile.artifact_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    wan_spec_with_artifact = wan_template.render(wan_profile, req)
    assert (
        "python3 -m inferweave.workers.wan --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers --port 8000"
        in wan_spec_with_artifact.run_command
    )
    assert "--quantize-4bit" in wan_spec_with_artifact.run_command


def test_worker_module_import_does_not_load_control_plane_or_httpx():
    """Validates that importing inferweave workers does not eagerly load control-plane or httpx."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import inferweave.workers.base\n"
        "assert 'httpx' not in sys.modules, f'httpx leaked into worker: {sys.modules.keys()}'\n"
        "assert 'inferweave.sdk' not in sys.modules, 'inferweave.sdk leaked into worker'\n"
        "assert 'inferweave.adapters.healthcheck' not in sys.modules, 'healthcheck adapter leaked'\n"
        "# Ensure public SDK imports work transparently via PEP 562 lazy loading\n"
        "from inferweave import InferWeave, DeploymentStatus, AutostopAction\n"
        "assert InferWeave is not None\n"
        "assert DeploymentStatus is not None\n"
        "assert AutostopAction is not None\n"
        "print('ISOLATION_OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "ISOLATION_OK" in res.stdout

