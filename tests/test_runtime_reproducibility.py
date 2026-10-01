"""Tests verifying reproducibility of built-in runtime templates.

Guarantees that built-in templates never regress to mutable `:latest` tags, unconstrained
`pip install -U` commands, or version ranges: built-in dependencies are pinned exactly.
"""

import re
import shlex

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.runtimes import manifest
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


_EXACT_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][A-Za-z0-9.+!_-]*$")


def test_builtin_diffusers_and_video_templates_pin_all_dependencies():
    """Policy: every package installed by built-in runtimes is pinned exactly (name==version)."""
    request = DeploymentRequest(model="test-model")
    for template_cls in (FluxDiffusersTemplate, WanVideoTemplate):
        template = template_cls()
        profile = _make_dummy_profile(WorkloadType.IMAGE, template.name)
        spec = template.render(profile, request)

        (cmd,) = spec.setup_commands
        tokens = shlex.split(cmd)
        assert tokens[:2] == ["pip", "install"]
        for pkg_spec in tokens[2:]:
            assert _EXACT_PIN.match(pkg_spec), (
                f"Template '{template.name}' installs non-exact spec '{pkg_spec}' in: {cmd}"
            )


def test_manifest_specs_are_exact_pins():
    spec_names = [n for n in dir(manifest) if n.endswith("_SPEC")]
    assert spec_names, "manifest exposes no *_SPEC constants"
    for name in spec_names:
        assert _EXACT_PIN.match(getattr(manifest, name)), f"{name} is not an exact pin"
    for pin in manifest.DIFFUSION_TRANSITIVE_PINS:
        assert _EXACT_PIN.match(pin), f"transitive pin {pin!r} is not exact"


def test_manifest_images_have_explicit_non_latest_tags():
    for name in ("VLLM_IMAGE", "FISH_SPEECH_IMAGE", "FISH_SPEECH_S2_IMAGE", "PYTORCH_IMAGE"):
        image = getattr(manifest, name)
        _, _, tag = image.rpartition(":")
        assert tag and tag != "latest" and "/" not in tag, f"{name}={image}"


def test_transitive_pins_are_unique_and_do_not_conflict_with_direct_specs():
    names = [p.split("==")[0].lower().replace("_", "-") for p in manifest.DIFFUSION_TRANSITIVE_PINS]
    assert len(names) == len(set(names)), "duplicate package in DIFFUSION_TRANSITIVE_PINS"
    direct = {
        getattr(manifest, n).split("==")[0].lower()
        for n in dir(manifest)
        if n.endswith("_SPEC")
    }
    assert not direct & set(names)
    # torch comes from PYTORCH_IMAGE; reinstalling it would replace the CUDA build.
    assert not {"torch", "triton"} & set(names)
    assert not any(n.startswith("nvidia-") for n in names)


def test_numpy_stays_on_1x_for_torch_2_4():
    """torch 2.4.0 wheels fail to initialise NumPy 2.x ('_ARRAY_API not found')."""
    (numpy_pin,) = [p for p in manifest.DIFFUSION_TRANSITIVE_PINS if p.startswith("numpy==")]
    assert numpy_pin.split("==")[1].startswith("1.")
    assert manifest.PYTORCH_IMAGE.startswith("pytorch/pytorch:2.4.0-")


def test_wan_runtime_installs_diffusers_with_wan_pipeline():
    """WanPipeline first shipped in diffusers 0.33.0."""
    from packaging.version import Version

    version = Version(manifest.DIFFUSERS_SPEC.split("==")[1])
    assert version >= Version("0.33.0")


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
