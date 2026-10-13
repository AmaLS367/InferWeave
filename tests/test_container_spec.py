"""Unit tests for ContainerSpec and ModelProfile container support (T1.1).

Validates:
1. ContainerSpec valid configurations (tags, digests, boundary ports, python versions).
2. ContainerSpec invalid configurations (empty image, shell string args, invalid ports, boolean port).
3. Credential rejection: platform credentials rejected in ContainerSpec.env and ModelProfile.
4. Factory construction via ModelProfile.for_container.
5. Backward compatibility with legacy ModelProfile JSON lacking container field.
"""

import json

import pytest
from pydantic import ValidationError

from inferweave import (
    ContainerSpec,
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
    WorkloadType,
)

# ---------------------------------------------------------------------------
# 1. ContainerSpec Valid Configurations
# ---------------------------------------------------------------------------


def test_container_spec_defaults():
    spec = ContainerSpec(image="vllm/vllm-openai:latest")
    assert spec.image == "vllm/vllm-openai:latest"
    assert spec.args == []
    assert spec.port == 8000
    assert spec.setup_commands == []
    assert spec.env == {}
    assert spec.add_python is None


def test_container_spec_with_sha256_digest():
    digest = "vllm/vllm-openai@sha256:45b23d06496e983443b4ff5f707f9f423806046399d50f16fbdb4094f43f972b"
    spec = ContainerSpec(image=digest)
    assert spec.image == digest


def test_container_spec_full_configuration():
    spec = ContainerSpec(
        image="ghcr.io/org/custom-server:v1.2.3",
        args=["--host", "0.0.0.0", "--port", "8080", "--workers", "4"],
        port=8080,
        setup_commands=["pip install packaging", "mkdir -p /tmp/cache"],
        env={"MODEL_DIR": "/models", "LOG_LEVEL": "DEBUG"},
        add_python="3.12",
    )
    assert spec.image == "ghcr.io/org/custom-server:v1.2.3"
    assert spec.args == ["--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
    assert spec.port == 8080
    assert spec.setup_commands == ["pip install packaging", "mkdir -p /tmp/cache"]
    assert spec.env == {"MODEL_DIR": "/models", "LOG_LEVEL": "DEBUG"}
    assert spec.add_python == "3.12"


@pytest.mark.parametrize("port", [1, 80, 443, 8000, 8080, 65535])
def test_container_spec_valid_boundary_ports(port):
    spec = ContainerSpec(image="alpine:latest", port=port)
    assert spec.port == port


@pytest.mark.parametrize("py_ver", ["3", "3.10", "3.11", "3.12", "3.12.2"])
def test_container_spec_valid_python_versions(py_ver):
    spec = ContainerSpec(image="ubuntu:22.04", add_python=py_ver)
    assert spec.add_python == py_ver


# ---------------------------------------------------------------------------
# 2. ContainerSpec Invalid Configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_image", ["", "   ", "image with spaces:latest"])
def test_container_spec_rejects_invalid_image(bad_image):
    with pytest.raises((ValidationError, ValueError)):
        ContainerSpec(image=bad_image)


@pytest.mark.parametrize(
    "bad_port", [0, -1, 65536, 70000, "not-a-port", 8000.5, True, False]
)
def test_container_spec_rejects_invalid_port(bad_port):
    with pytest.raises((ValidationError, ValueError, TypeError)):
        ContainerSpec(image="alpine:latest", port=bad_port)


def test_container_spec_rejects_shell_command_args_string():
    """args must be a list of argv tokens, never a raw shell command string."""
    with pytest.raises((ValidationError, ValueError)) as excinfo:
        ContainerSpec(image="python:3.11", args="python -m http.server 8000")
    assert "argv" in str(excinfo.value).lower() or "list" in str(excinfo.value).lower()


@pytest.mark.parametrize("bad_args", [123, ["python", 123], [None], [["nested"]]])
def test_container_spec_rejects_non_string_args_elements(bad_args):
    with pytest.raises((ValidationError, ValueError, TypeError)):
        ContainerSpec(image="python:3.11", args=bad_args)


def test_container_spec_rejects_shell_command_setup_string():
    with pytest.raises((ValidationError, ValueError)):
        ContainerSpec(
            image="alpine", setup_commands="apt-get update && apt-get install -y git"
        )


@pytest.mark.parametrize(
    "bad_py", ["", "   ", "python3.12", "invalid-version", "3.12-rc1"]
)
def test_container_spec_rejects_invalid_python_version(bad_py):
    with pytest.raises((ValidationError, ValueError)):
        ContainerSpec(image="alpine", add_python=bad_py)


# ---------------------------------------------------------------------------
# 3. Credential Rejection
# ---------------------------------------------------------------------------


PLATFORM_CREDENTIAL_KEYS = [
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_PROXY_TOKEN_ID",
    "MODAL_PROXY_TOKEN_SECRET",
    "RUNPOD_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "API_KEY",
    "MY_API_KEY",
    "SECRET_KEY",
    "PASSWORD",
    "AUTH_TOKEN",
    "AUTHORIZATION",
    "BEARER_TOKEN",
    "CREDENTIALS",
    "SERVICE_AUTH",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
]


@pytest.mark.parametrize("cred_key", PLATFORM_CREDENTIAL_KEYS)
def test_container_spec_env_rejects_platform_credentials(cred_key):
    with pytest.raises((ValidationError, ValueError)) as excinfo:
        ContainerSpec(image="alpine:latest", env={cred_key: "leaked_secret_val"})
    assert (
        "forbidden" in str(excinfo.value).lower()
        or "credential" in str(excinfo.value).lower()
    )


@pytest.mark.parametrize("cred_key", PLATFORM_CREDENTIAL_KEYS)
def test_for_container_rejects_platform_credentials_in_env(cred_key):
    with pytest.raises((ValidationError, ValueError)):
        ModelProfile.for_container(
            id="test-model",
            image="alpine:latest",
            env={cred_key: "leaked_secret_val"},
        )


@pytest.mark.parametrize("cred_key", PLATFORM_CREDENTIAL_KEYS)
def test_for_container_rejects_platform_credentials_in_default_env(cred_key):
    with pytest.raises((ValidationError, ValueError)):
        ModelProfile.for_container(
            id="test-model",
            image="alpine:latest",
            default_env={cred_key: "leaked_secret_val"},
        )


@pytest.mark.parametrize("cred_key", PLATFORM_CREDENTIAL_KEYS)
def test_model_profile_rejects_credentials_in_default_env(cred_key):
    with pytest.raises((ValidationError, ValueError)):
        ModelProfile(
            id="test-model",
            name="Test Model",
            workload_type=WorkloadType.CUSTOM,
            default_runtime="container",
            hardware=HardwareRequirements(min_vram_gb=16),
            default_env={cred_key: "secret"},
            container=ContainerSpec(image="alpine"),
        )


def test_container_spec_permits_nonsecret_environment_variables():
    allowed_env = {
        "PORT": "8080",
        "HOST": "0.0.0.0",
        "DEBUG": "false",
        "WORKERS": "4",
        "MODEL_PATH": "/data/weights",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHONUNBUFFERED": "1",
    }
    spec = ContainerSpec(image="alpine", env=allowed_env)
    assert spec.env == allowed_env


# ---------------------------------------------------------------------------
# 4. Factory Construction (ModelProfile.for_container)
# ---------------------------------------------------------------------------


def test_for_container_positional_arguments():
    profile = ModelProfile.for_container(
        "my-classifier",
        "my-registry.io/classifier:v2.0",
        ["serve", "--port", "9000"],
        9000,
        "/ready",
        12.0,
    )
    assert profile.id == "my-classifier"
    assert profile.name == "my-classifier"
    assert profile.workload_type == WorkloadType.CUSTOM
    assert profile.default_runtime == "container"
    assert profile.source == "container"
    assert profile.hardware.min_vram_gb == 12.0
    assert profile.hardware.gpu_count == 1
    assert profile.healthcheck.port == 9000
    assert profile.healthcheck.path == "/ready"
    assert profile.container is not None
    assert profile.container.image == "my-registry.io/classifier:v2.0"
    assert profile.container.args == ["serve", "--port", "9000"]
    assert profile.container.port == 9000


def test_for_container_keyword_arguments_and_overrides():
    profile = ModelProfile.for_container(
        id="custom-embed",
        image="embedder:latest",
        name="Custom Embedding Server",
        args=["python", "-m", "server"],
        port=8080,
        health_path="/v1/health",
        min_vram_gb=8.0,
        setup_commands=["pip install torch"],
        env={"MODEL_NAME": "bge-large"},
        add_python="3.11",
        gpu_count=2,
        recommended_gpus=["L4", "A10G"],
        min_cpu_cores=8,
        min_ram_gb=32.0,
    )
    assert profile.id == "custom-embed"
    assert profile.name == "Custom Embedding Server"
    assert profile.workload_type == WorkloadType.CUSTOM
    assert profile.default_runtime == "container"
    assert profile.hardware.min_vram_gb == 8.0
    assert profile.hardware.gpu_count == 2
    assert profile.hardware.recommended_gpus == ["L4", "A10G"]
    assert profile.hardware.min_cpu_cores == 8
    assert profile.hardware.min_ram_gb == 32.0
    assert profile.healthcheck.port == 8080
    assert profile.healthcheck.path == "/v1/health"
    assert profile.container is not None
    assert profile.container.setup_commands == ["pip install torch"]
    assert profile.container.env == {"MODEL_NAME": "bge-large"}
    assert profile.container.add_python == "3.11"


def test_for_container_explicit_hardware_and_healthcheck():
    hw = HardwareRequirements(min_vram_gb=24.0, gpu_count=2, recommended_gpus=["A100"])
    hc = HealthcheckConfig(port=8888, path="/ping", timeout_seconds=120.0)
    profile = ModelProfile.for_container(
        id="explicit-hw",
        image="app:v1",
        hardware=hw,
        healthcheck=hc,
    )
    assert profile.hardware is hw
    assert profile.healthcheck is hc
    assert profile.container.port == 8000


# ---------------------------------------------------------------------------
# 5. Legacy JSON Compatibility and Round-Trip Serialization
# ---------------------------------------------------------------------------


def test_loading_legacy_model_profile_json_without_container():
    """0.3.x ModelProfile JSON documents have no 'container' field and must load cleanly."""
    legacy_json = json.dumps(
        {
            "id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "name": "Llama 3 8B Instruct",
            "workload_type": "llm",
            "default_runtime": "vllm",
            "source": "huggingface",
            "artifact_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "hardware": {
                "min_vram_gb": 16.0,
                "recommended_gpus": ["A10G", "L4", "A100"],
                "gpu_count": 1,
                "min_cpu_cores": 4,
                "min_ram_gb": 16.0,
                "min_cuda_version": "12.1",
            },
            "healthcheck": {
                "enabled": True,
                "path": "/health",
                "port": 8000,
                "initial_delay_seconds": 10.0,
                "timeout_seconds": 300.0,
                "probe_interval_seconds": 3.0,
                "request_timeout_seconds": 5.0,
                "expected_status_codes": [200],
                "consecutive_successes": 1,
                "method": "GET",
                "headers": {},
            },
            "default_env": {},
            "metadata": {},
        }
    )

    profile = ModelProfile.model_validate_json(legacy_json)
    assert profile.id == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert profile.container is None
    assert profile.workload_type == WorkloadType.LLM
    assert profile.hardware.min_vram_gb == 16.0


def test_round_trip_serialization_with_container():
    profile = ModelProfile.for_container(
        id="round-trip-test",
        image="repo/img@sha256:45b23d06496e983443b4ff5f707f9f423806046399d50f16fbdb4094f43f972b",
        args=["--flag", "value"],
        port=9999,
        health_path="/status",
        min_vram_gb=10.0,
        env={"CUSTOM_KEY": "custom_val"},
        add_python="3.12",
    )
    dumped = profile.model_dump_json()
    loaded = ModelProfile.model_validate_json(dumped)

    assert loaded.id == profile.id
    assert loaded.container is not None
    assert loaded.container.image == profile.container.image
    assert loaded.container.args == ["--flag", "value"]
    assert loaded.container.port == 9999
    assert loaded.container.env == {"CUSTOM_KEY": "custom_val"}
    assert loaded.container.add_python == "3.12"
    assert loaded.healthcheck.port == 9999
    assert loaded.healthcheck.path == "/status"
