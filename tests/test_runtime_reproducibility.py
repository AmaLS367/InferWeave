"""Tests verifying reproducibility of built-in runtime templates.

Guarantees that built-in templates never regress to mutable image tags (they must use
immutable ``@sha256:`` digests; custom runtimes may still use tags), unconstrained
`pip install -U` commands, or version ranges: built-in dependencies are pinned exactly.
"""

import re
import shlex
from unittest.mock import patch

import pytest

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
    FishSpeechS2Template,
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


_IMMUTABLE_IMAGE = re.compile(r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
_MANIFEST_IMAGE_NAMES = ("VLLM_IMAGE", "FISH_SPEECH_IMAGE", "FISH_SPEECH_S2_IMAGE", "PYTORCH_IMAGE")


def test_immutable_image_pattern_rejects_mutable_references():
    """Guards the guard: tag-only references must fail the reproducibility check."""
    digest = "sha256:" + "a" * 64
    assert _IMMUTABLE_IMAGE.match(f"vllm/vllm-openai@{digest}")
    for mutable in (
        "vllm/vllm-openai:v0.7.3",
        "vllm/vllm-openai:latest",
        "vllm/vllm-openai",
        f"vllm/vllm-openai:v0.7.3@{digest}",
        "vllm/vllm-openai@sha256:abc123",
        f"vllm/vllm-openai@sha1:{'a' * 40}",
    ):
        assert not _IMMUTABLE_IMAGE.match(mutable), mutable


def test_builtin_templates_use_immutable_image_digests():
    """Every built-in production runtime must reference its image by sha256 digest, not a tag."""
    request = DeploymentRequest(model="test-model")
    pinned = {getattr(manifest, n) for n in _MANIFEST_IMAGE_NAMES}
    for name, template in _BUILTIN_TEMPLATES.items():
        profile = _make_dummy_profile(WorkloadType.LLM, name)
        spec = template.render(profile, request)

        assert _IMMUTABLE_IMAGE.match(spec.docker_image), (
            f"Template '{name}' image is not digest-pinned (repo@sha256:<64 hex>): "
            f"{spec.docker_image}"
        )
        assert spec.docker_image in pinned, (
            f"Template '{name}' bypasses the manifest image pins: {spec.docker_image}"
        )


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


def test_manifest_images_are_immutable_digests_with_documented_source_tag():
    images = [getattr(manifest, name) for name in _MANIFEST_IMAGE_NAMES]
    assert len(set(images)) == len(images), "built-in runtimes must not share a digest by accident"
    for name in _MANIFEST_IMAGE_NAMES:
        image = getattr(manifest, name)
        assert _IMMUTABLE_IMAGE.match(image), f"{name}={image} is not repo@sha256:<64 hex>"

        # The human-readable upstream tag the digest was resolved from is kept as metadata.
        source = manifest.BUILTIN_IMAGE_TAGS[image]
        repo, _, tag = source.rpartition(":")
        assert repo == image.split("@")[0], f"{name}: source tag {source!r} is for another repo"
        assert tag and tag != "latest", f"{name}: source tag {source!r} must be a version tag"
    assert set(manifest.BUILTIN_IMAGE_TAGS) == set(images)


def test_fish_s2_runtime_is_documented_as_beta():
    """Upstream ships the S2 runtime as the v2.0.0-beta pre-release; keep that visible."""
    assert manifest.BUILTIN_IMAGE_TAGS[manifest.FISH_SPEECH_S2_IMAGE].endswith("v2.0.0-beta")
    assert "Beta" in (FishSpeechS2Template.__doc__ or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("image_name", _MANIFEST_IMAGE_NAMES)
async def test_modal_receives_builtin_digest_reference_unchanged(image_name):
    """Modal consumes the digest form: it must reach Image.from_registry verbatim."""
    import modal
    from support import modal_record

    from inferweave.accounts import ProviderAccount
    from inferweave.providers.modal_provider import ModalProvider

    image = getattr(manifest, image_name)
    runtime = RuntimeSpec(name="pinned", docker_image=image, run_command="true", port=8000)
    profile = _make_dummy_profile(WorkloadType.LLM, "pinned")
    request = DeploymentRequest(model=profile.id, provider="modal")
    provider = ModalProvider()
    account = ProviderAccount.ambient("modal")
    record = modal_record(provider, profile, account, endpoint_url=None)

    real_from_registry = modal.Image.from_registry
    with (
        patch("modal.Image.from_registry", side_effect=real_from_registry) as from_registry,
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://pinned.modal.run"),
    ):
        await provider.provision(record, request, profile, runtime, account)

    from_registry.assert_called_once()
    assert from_registry.call_args.args[0] == image


@pytest.mark.parametrize("image_name", _MANIFEST_IMAGE_NAMES)
def test_skypilot_receives_builtin_digest_reference_unchanged(image_name):
    """SkyPilot consumes the digest form: image_id must be exactly ``docker:<repo>@sha256:...``."""
    from inferweave.accounts import ProviderAccount
    from inferweave.domain.deployment_record import DeploymentRecord
    from inferweave.providers.skypilot import SkyPilotProvider

    image = getattr(manifest, image_name)
    runtime = RuntimeSpec(name="pinned", docker_image=image, run_command="true", port=8000)
    profile = _make_dummy_profile(WorkloadType.LLM, "pinned")
    request = DeploymentRequest(model=profile.id, provider="runpod", gpu_type="A100")

    provider = SkyPilotProvider(cloud_name="runpod")
    deployment_id = provider.new_deployment_id(profile)
    record = DeploymentRecord(
        id=deployment_id,
        model=profile.id,
        provider="runpod",
        resource=provider.resource_ref(deployment_id, request, ProviderAccount.ambient("runpod")),
    )
    payload = provider.launch_payload(record, request, profile, runtime)

    assert payload["resources"]["image_id"] == f"docker:{image}"
    assert payload["cluster"] == deployment_id


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
    assert manifest.BUILTIN_IMAGE_TAGS[manifest.PYTORCH_IMAGE].startswith("pytorch/pytorch:2.4.0-")


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
