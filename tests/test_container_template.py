"""Unit and regression tests for ContainerTemplate and RuntimeSpec integration (T1.2).

Validates:
1. Template registration: ContainerTemplate is named 'container', registered in _BUILTIN_TEMPLATES,
   and accessible via get_runtime_template("container").
2. Rendering valid configurations: RuntimeSpec correctly preserves image, argv, setup commands,
   port, environment, hardware requirements, and healthcheck configuration.
3. Missing configuration: raises ValueError when profile lacks container specification.
4. Environment merging and validation: verifies precedence and rejection of sensitive credentials
   or non-string types.
5. Argv boundaries and command hardening: verifies discrete argv token preservation and absence
   of shell injection.
6. Modal compatibility: ensures add_python is preserved and accessible on RuntimeSpec and metadata.
"""

import shlex

import pytest

from inferweave.domain.options import DeploymentOptions, RuntimeOptions
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    ContainerSpec,
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.runtimes.base import RuntimeSpec, RuntimeTemplate
from inferweave.runtimes.templates import (
    _BUILTIN_TEMPLATES,
    ContainerTemplate,
    get_runtime_template,
)

# ---------------------------------------------------------------------------
# 1. Template Registration
# ---------------------------------------------------------------------------


def test_container_template_registration():
    """'container' is registered in _BUILTIN_TEMPLATES and accessible via get_runtime_template()."""
    template = get_runtime_template("container")
    assert isinstance(template, ContainerTemplate)
    assert isinstance(template, RuntimeTemplate)
    assert template.name == "container"
    assert "container" in _BUILTIN_TEMPLATES
    assert _BUILTIN_TEMPLATES["container"] is template


def test_get_runtime_template_unknown_lists_container():
    """When an unknown template is requested, 'container' is listed among available templates."""
    with pytest.raises(ValueError) as excinfo:
        get_runtime_template("nonexistent-runtime")
    error_msg = str(excinfo.value)
    assert "container" in error_msg
    assert "vllm" in error_msg


# ---------------------------------------------------------------------------
# 2. Rendering Valid Configurations
# ---------------------------------------------------------------------------


def test_render_minimal_container():
    """Renders a minimal container profile with default settings."""
    profile = ModelProfile.for_container(
        id="test-minimal",
        image="ghcr.io/org/service:v1.0.0",
    )
    request = DeploymentRequest(model="test-minimal")
    template = get_runtime_template("container")

    spec = template.render(profile, request)

    assert isinstance(spec, RuntimeSpec)
    assert spec.name == "container"
    assert spec.docker_image == "ghcr.io/org/service:v1.0.0"
    assert spec.run_args == []
    assert spec.run_command == ""
    assert spec.setup_commands == []
    assert spec.port == 8000
    assert spec.env_vars == {}
    assert spec.healthcheck_path == "/health"
    assert spec.add_python is None
    assert spec.hardware == profile.hardware
    assert spec.healthcheck == profile.healthcheck


def test_render_image_with_digest():
    """Preserves immutable sha256 digest container image reference."""
    digest = "myrepo/worker@sha256:45b23d06496e983443b4ff5f707f9f423806046399d50f16fbdb4094f43f972b"
    profile = ModelProfile.for_container(id="test-digest", image=digest)
    template = ContainerTemplate()
    spec = template.render(profile, DeploymentRequest(model="test-digest"))

    assert spec.docker_image == digest


def test_render_full_container_configuration():
    """Renders a fully-specified container profile preserving all attributes."""
    hw = HardwareRequirements(
        min_vram_gb=24.0,
        gpu_count=2,
        recommended_gpus=["A100", "L40S"],
        min_cpu_cores=8,
        min_ram_gb=32.0,
        min_cuda_version="12.2",
    )
    hc = HealthcheckConfig(
        path="/v1/status",
        port=9000,
        initial_delay_seconds=15.0,
        timeout_seconds=180.0,
        probe_interval_seconds=4.0,
        expected_status_codes=[200, 204],
        method="GET",
        headers={"X-Service": "inferweave"},
    )
    profile = ModelProfile.for_container(
        id="test-full",
        image="custom/runner:2.1.0",
        args=["python", "-m", "worker.main", "--workers", "4"],
        port=9000,
        setup_commands=["pip install uv", "uv venv /app/.venv"],
        env={"MODEL_DIR": "/models", "LOG_LEVEL": "INFO"},
        add_python="3.12",
        hardware=hw,
        healthcheck=hc,
        metadata={"custom_tag": "prod-east"},
    )
    request = DeploymentRequest(model="test-full")
    template = ContainerTemplate()

    spec = template.render(profile, request)

    assert spec.name == "container"
    assert spec.docker_image == "custom/runner:2.1.0"
    assert spec.run_args == ["python", "-m", "worker.main", "--workers", "4"]
    assert spec.run_command == "python -m worker.main --workers 4"
    assert spec.setup_commands == ["pip install uv", "uv venv /app/.venv"]
    assert spec.port == 9000
    assert spec.env_vars == {"MODEL_DIR": "/models", "LOG_LEVEL": "INFO"}
    assert spec.healthcheck_path == "/v1/status"
    assert spec.add_python == "3.12"
    assert spec.metadata.get("add_python") == "3.12"
    assert spec.metadata.get("custom_tag") == "prod-east"
    assert spec.hardware == hw
    assert spec.healthcheck == hc
    assert spec.metadata.get("hardware") == hw.model_dump()
    assert spec.metadata.get("healthcheck") == hc.model_dump()


# ---------------------------------------------------------------------------
# 3. Missing Configuration
# ---------------------------------------------------------------------------


def test_render_missing_container_specification():
    """Raises ValueError when ModelProfile has default_runtime='container' but container is None."""
    profile = ModelProfile(
        id="orphan-container",
        name="Orphan Container Model",
        workload_type=WorkloadType.CUSTOM,
        default_runtime="container",
        hardware=HardwareRequirements(min_vram_gb=16, gpu_count=1),
        healthcheck=HealthcheckConfig(port=8000),
        container=None,
    )
    request = DeploymentRequest(model="orphan-container")
    template = get_runtime_template("container")

    with pytest.raises(ValueError) as excinfo:
        template.render(profile, request)

    assert "missing" in str(excinfo.value).lower()
    assert "container" in str(excinfo.value).lower()
    assert "orphan-container" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Environment Merging and Validation
# ---------------------------------------------------------------------------


def test_env_merging_precedence():
    """Validates env precedence: container.env < profile.default_env < request.env < extra_env."""
    profile = ModelProfile(
        id="env-prec",
        name="Env Precedence Test",
        workload_type=WorkloadType.CUSTOM,
        default_runtime="container",
        hardware=HardwareRequirements(min_vram_gb=8, gpu_count=1),
        healthcheck=HealthcheckConfig(port=8000),
        container=ContainerSpec(
            image="alpine:latest",
            env={"A": "from-container", "OVERRIDDEN": "from-container"},
        ),
        default_env={"B": "from-default", "OVERRIDDEN": "from-default"},
    )
    request = DeploymentRequest(
        model="env-prec",
        env={"C": "from-request", "OVERRIDDEN": "from-request"},
        options=DeploymentOptions(
            runtime=RuntimeOptions(
                extra_env={"D": "from-extra", "OVERRIDDEN": "from-extra"},
            )
        ),
    )
    template = ContainerTemplate()
    spec = template.render(profile, request)

    assert spec.env_vars["A"] == "from-container"
    assert spec.env_vars["B"] == "from-default"
    assert spec.env_vars["C"] == "from-request"
    assert spec.env_vars["D"] == "from-extra"
    assert spec.env_vars["OVERRIDDEN"] == "from-extra"


@pytest.mark.parametrize(
    "cred_key",
    [
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "LIGHTNING_API_KEY",
        "RUNPOD_API_KEY",
        "VAST_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "MY_AUTH_TOKEN",
        "SERVICE_API_KEY",
        "BEARER_TOKEN",
        "APP_PASSWORD",
    ],
)
def test_render_rejects_credentials_in_request_env(cred_key):
    """Rejects platform credentials or sensitive keys passed in DeploymentRequest.env."""
    profile = ModelProfile.for_container(id="sec-check", image="alpine:latest")
    request = DeploymentRequest(
        model="sec-check",
        env={cred_key: "leaked_secret_value"},
    )
    template = ContainerTemplate()

    with pytest.raises(ValueError) as excinfo:
        template.render(profile, request)

    assert "forbidden" in str(excinfo.value).lower()
    assert cred_key in str(excinfo.value)


@pytest.mark.parametrize(
    "cred_key",
    [
        "MODAL_TOKEN_ID",
        "LIGHTNING_API_KEY",
        "SECRET_KEY",
        "API_KEY",
    ],
)
def test_render_rejects_credentials_in_extra_env(cred_key):
    """Rejects platform credentials or sensitive keys passed in RuntimeOptions.extra_env."""
    profile = ModelProfile.for_container(id="sec-extra", image="alpine:latest")
    request = DeploymentRequest(
        model="sec-extra",
        options=DeploymentOptions(
            runtime=RuntimeOptions(
                extra_env={cred_key: "leaked_extra_secret"},
            )
        ),
    )
    template = ContainerTemplate()

    with pytest.raises(ValueError) as excinfo:
        template.render(profile, request)

    assert "forbidden" in str(excinfo.value).lower()
    assert cred_key in str(excinfo.value)


@pytest.mark.parametrize(
    "bad_env_key",
    [123, None, ("tuple",)],
)
def test_render_rejects_non_string_env_keys(bad_env_key):
    """Rejects non-string environment variable keys."""
    profile = ModelProfile.for_container(id="bad-env-k", image="alpine:latest")
    request = DeploymentRequest(model="bad-env-k")
    # Force invalid key directly into request.env (bypassing pydantic if necessary)
    request.env[bad_env_key] = "val"
    template = ContainerTemplate()

    with pytest.raises(TypeError) as excinfo:
        template.render(profile, request)

    assert "string" in str(excinfo.value).lower()


@pytest.mark.parametrize(
    "bad_env_val",
    [123, None, ["list"], {"dict": "val"}],
)
def test_render_rejects_non_string_env_values(bad_env_val):
    """Rejects non-string environment variable values."""
    profile = ModelProfile.for_container(id="bad-env-v", image="alpine:latest")
    request = DeploymentRequest(model="bad-env-v")
    request.env["TEST_KEY"] = bad_env_val
    template = ContainerTemplate()

    with pytest.raises(TypeError) as excinfo:
        template.render(profile, request)

    assert "string" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# 5. Argv Boundaries and Shell Injection Prevention
# ---------------------------------------------------------------------------


ADVERSARIAL_CONTAINER_ARGS = [
    ["; touch /tmp/pwned", "--flag"],
    ["--path", "path with spaces/dir", "$(rm -rf /)"],
    ["`echo pwned`", "| cat /etc/passwd", "&& id"],
    ["payload 'with' \"quotes\" and 'special' characters", "> /tmp/out"],
    ["--option=key:value;dangerous", "newline\narg", "tab\targ"],
]


@pytest.mark.parametrize("args_payload", ADVERSARIAL_CONTAINER_ARGS)
def test_argv_boundaries_preserved_without_shell_injection(args_payload):
    """Argv tokens are preserved literally; shlex.split(run_command) recovers original tokens."""
    profile = ModelProfile.for_container(
        id="adv-container",
        image="alpine:latest",
        args=args_payload,
    )
    request = DeploymentRequest(model="adv-container")
    template = ContainerTemplate()

    spec = template.render(profile, request)

    # 1. run_args must be an exact match of discrete argv tokens
    assert spec.run_args == args_payload

    # 2. Parsing the shell run_command with shlex.split recovers the exact same tokens
    recovered = shlex.split(spec.run_command)
    assert recovered == args_payload


def test_runtime_options_extra_cli_args_appended_to_argv():
    """Extra CLI args and engine args from request options are safely appended as argv tokens."""
    profile = ModelProfile.for_container(
        id="cli-opts",
        image="vllm/vllm-openai:latest",
        args=["--host", "0.0.0.0"],
    )
    request = DeploymentRequest(
        model="cli-opts",
        custom_args={
            "extra_cli_args": [
                "--chat-template",
                "{% for m in messages %}{{ m }}{% endfor %}",
            ],
            "runtime_args": {
                "engine_args": {"max_model_len": 4096, "trust_remote_code": True},
            },
        },
    )
    template = ContainerTemplate()
    spec = template.render(profile, request)

    assert spec.run_args[:2] == ["--host", "0.0.0.0"]
    assert "--max-model-len" in spec.run_args
    assert "4096" in spec.run_args
    assert "--trust-remote-code" in spec.run_args
    assert "{% for m in messages %}{{ m }}{% endfor %}" in spec.run_args
    assert shlex.split(spec.run_command) == spec.run_args


# ---------------------------------------------------------------------------
# 6. Modal Integration Compatibility (add_python)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_ver", ["3.10", "3.11", "3.12", "3.12.3"])
def test_modal_add_python_preservation(py_ver):
    """Ensures add_python is preserved and accessible for the later Modal integration."""
    profile = ModelProfile.for_container(
        id="modal-py",
        image="nvidia/cuda:12.2.0-base-ubuntu22.04",
        add_python=py_ver,
    )
    template = ContainerTemplate()
    spec = template.render(profile, DeploymentRequest(model="modal-py"))

    # Accessible on RuntimeSpec directly
    assert spec.add_python == py_ver
    # Accessible via metadata dictionary
    assert spec.metadata.get("add_python") == py_ver
    # Accessible on ModelProfile.container
    assert profile.container.add_python == py_ver


def test_modal_add_python_none_when_unspecified():
    """add_python defaults to None when unspecified."""
    profile = ModelProfile.for_container(id="no-py", image="python:3.11-slim")
    template = ContainerTemplate()
    spec = template.render(profile, DeploymentRequest(model="no-py"))

    assert spec.add_python is None
    assert spec.metadata.get("add_python") is None
    assert profile.container.add_python is None


# ---------------------------------------------------------------------------
# 7. Readiness Configuration (HealthcheckConfig)
# ---------------------------------------------------------------------------


def test_readiness_healthcheck_preservation():
    """Verifies that custom HealthcheckConfig is fully preserved in RuntimeSpec."""
    hc = HealthcheckConfig(
        enabled=True,
        path="/v2/health/ready",
        port=8080,
        initial_delay_seconds=20.0,
        timeout_seconds=600.0,
        probe_interval_seconds=5.0,
        request_timeout_seconds=10.0,
        expected_status_codes=[200, 204],
        consecutive_successes=2,
        method="HEAD",
        headers={"Authorization": "Bearer internal-health-key"},
    )
    profile = ModelProfile.for_container(
        id="custom-hc",
        image="my-service:latest",
        port=8080,
        healthcheck=hc,
    )
    template = ContainerTemplate()
    spec = template.render(profile, DeploymentRequest(model="custom-hc"))

    assert spec.port == 8080
    assert spec.healthcheck_path == "/v2/health/ready"
    assert spec.healthcheck == hc
    assert spec.metadata.get("healthcheck") == hc.model_dump()
    assert spec.healthcheck.method == "HEAD"
    assert spec.healthcheck.consecutive_successes == 2
    assert spec.healthcheck.expected_status_codes == [200, 204]


# ---------------------------------------------------------------------------
# 8. Hardware Requirements Preservation
# ---------------------------------------------------------------------------


def test_hardware_requirements_preservation():
    """Verifies that HardwareRequirements is preserved in RuntimeSpec."""
    hw = HardwareRequirements(
        min_vram_gb=80.0,
        gpu_count=4,
        recommended_gpus=["H100", "A100"],
        min_cpu_cores=16,
        min_ram_gb=128.0,
        min_cuda_version="12.4",
    )
    profile = ModelProfile.for_container(
        id="heavy-gpu",
        image="llm-cluster:latest",
        hardware=hw,
    )
    template = ContainerTemplate()
    spec = template.render(profile, DeploymentRequest(model="heavy-gpu"))

    assert spec.hardware == hw
    assert spec.metadata.get("hardware") == hw.model_dump()
    assert spec.hardware.min_vram_gb == 80.0
    assert spec.hardware.gpu_count == 4
    assert spec.hardware.recommended_gpus == ["H100", "A100"]
    assert spec.hardware.min_cuda_version == "12.4"
