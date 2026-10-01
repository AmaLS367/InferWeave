"""Tests verifying reproducibility of built-in runtime templates.

Guarantees that built-in templates never regress to mutable `:latest` tags or
unconstrained `pip install -U` commands, ensuring predictable deployments.
"""

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.runtimes.base import RuntimeSpec, RuntimeTemplate
from inferweave.runtimes.templates import (
    _BUILTIN_TEMPLATES,
    FluxDiffusersTemplate,
    WanVideoTemplate,
)


def _make_dummy_profile(workload: WorkloadType, runtime: str) -> ModelProfile:
    return ModelProfile(
        id="test-model",
        name="Test Model",
        workload_type=workload,
        default_runtime=runtime,
        hardware=HardwareRequirements(min_vram_gb=16, gpu_count=1),
        healthcheck=HealthcheckConfig(port=8000),
    )


def test_builtin_templates_have_no_latest_docker_tags():
    """Built-in templates must pin specific container tags and forbid :latest."""
    request = DeploymentRequest(model="test-model")
    for name, template in _BUILTIN_TEMPLATES.items():
        profile = _make_dummy_profile(WorkloadType.LLM, name)
        spec = template.render(profile, request)

        assert ":latest" not in spec.docker_image, (
            f"Template '{name}' uses mutable tag in image: {spec.docker_image}"
        )
        assert not spec.docker_image.endswith("latest"), (
            f"Template '{name}' ends with mutable 'latest': {spec.docker_image}"
        )
        # Must have a tag separator ':'
        assert ":" in spec.docker_image, f"Template '{name}' has unversioned image: {spec.docker_image}"


def test_builtin_templates_have_no_unconstrained_pip_upgrade():
    """Built-in templates must not execute 'pip install -U' without bounded versions."""
    request = DeploymentRequest(model="test-model")
    for name, template in _BUILTIN_TEMPLATES.items():
        profile = _make_dummy_profile(WorkloadType.LLM, name)
        spec = template.render(profile, request)

        for cmd in spec.setup_commands:
            assert "pip install -U" not in cmd, (
                f"Template '{name}' contains unconstrained 'pip install -U': {cmd}"
            )
            assert "pip install --upgrade" not in cmd, (
                f"Template '{name}' contains unconstrained 'pip install --upgrade': {cmd}"
            )


def test_builtin_diffusers_and_video_templates_pin_all_dependencies():
    """Dependencies in setup_commands must have explicit version bounds (==, >=, or <)."""
    request = DeploymentRequest(model="test-model")
    for template_cls in (FluxDiffusersTemplate, WanVideoTemplate):
        template = template_cls()
        profile = _make_dummy_profile(WorkloadType.IMAGE, template.name)
        spec = template.render(profile, request)

        for cmd in spec.setup_commands:
            if not cmd.startswith("pip install"):
                continue
            # Extract package specs after 'pip install'
            tokens = cmd.split()[2:]
            for pkg_spec in tokens:
                assert any(op in pkg_spec for op in ("==", ">=", "<=", "<", "~=")), (
                    f"Template '{template.name}' installs unversioned package '{pkg_spec}' in: {cmd}"
                )


def test_custom_user_defined_runtimes_permit_custom_images():
    """Custom templates may choose their own images or mutable configurations if desired."""

    class CustomTemplate(RuntimeTemplate):
        @property
        def name(self) -> str:
            return "custom"

        def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
            return RuntimeSpec(
                name=self.name,
                docker_image="my-org/custom-worker:latest",
                run_command="python server.py",
                port=8080,
            )

    custom = CustomTemplate()
    profile = _make_dummy_profile(WorkloadType.LLM, "custom")
    spec = custom.render(profile, DeploymentRequest(model="test-model"))
    assert spec.docker_image == "my-org/custom-worker:latest"
    assert spec.run_args == ["python", "server.py"]
