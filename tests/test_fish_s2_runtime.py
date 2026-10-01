"""Regression tests: the built-in ``fish-s2-pro`` profile must deploy a runtime that supports S2.

Upstream basis (fishaudio/fish-speech tag v2.0.0-beta, the first release with S2 Pro): weights via
``hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro``, server ``tools/api_server.py``
with ``--llama-checkpoint-path checkpoints/s2-pro --decoder-checkpoint-path
checkpoints/s2-pro/codec.pth --decoder-config-name modded_dac_vq --listen 0.0.0.0:8080``,
health endpoint ``GET /v1/health``, official image ``fishaudio/fish-speech:server-cuda-v2.0.0-beta``.
The v1.5.1 image cannot load S2 checkpoints.
"""

import shlex

import pytest

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.manifest import (
    FISH_S2_PRO_REVISION,
    FISH_SPEECH_IMAGE,
    FISH_SPEECH_S2_IMAGE,
)
from inferweave.runtimes.templates import (
    FishSpeechS2Template,
    FishSpeechTemplate,
    get_runtime_template,
)


@pytest.fixture
def s2_profile() -> ModelProfile:
    return ModelRegistry().get("fish-s2-pro")


def test_builtin_s2_profile_uses_s2_runtime(s2_profile):
    assert s2_profile.artifact_id == "fishaudio/s2-pro"
    assert s2_profile.default_runtime == "fish-speech-s2"
    assert isinstance(get_runtime_template(s2_profile.default_runtime), FishSpeechS2Template)


def test_s2_runtime_is_not_the_legacy_v1_image(s2_profile):
    spec = get_runtime_template(s2_profile.default_runtime).render(
        s2_profile, DeploymentRequest(model=s2_profile.id)
    )
    assert spec.docker_image == FISH_SPEECH_S2_IMAGE
    assert spec.docker_image == "fishaudio/fish-speech:server-cuda-v2.0.0-beta"
    assert spec.docker_image != FISH_SPEECH_IMAGE
    assert "v1." not in spec.docker_image


def test_s2_render_matches_upstream_deployment(s2_profile):
    spec = get_runtime_template(s2_profile.default_runtime).render(
        s2_profile, DeploymentRequest(model=s2_profile.id)
    )

    assert spec.name == "fish-speech-s2"
    assert spec.port == 8080
    assert spec.healthcheck_path == "/v1/health"

    # Checkpoint layout follows the HF repo name (checkpoints/s2-pro), not the profile id.
    assert spec.run_args == [
        "/app/.venv/bin/python",
        "/app/tools/api_server.py",
        "--listen",
        "0.0.0.0:8080",
        "--llama-checkpoint-path",
        "/app/checkpoints/s2-pro",
        "--decoder-checkpoint-path",
        "/app/checkpoints/s2-pro/codec.pth",
        "--decoder-config-name",
        "modded_dac_vq",
    ]
    assert shlex.split(spec.run_command) == spec.run_args

    # Weights are fetched from the S2 Pro repo at the pinned revision into the same directory.
    (setup,) = spec.setup_commands
    assert shlex.split(setup) == [
        "/app/.venv/bin/hf",
        "download",
        "fishaudio/s2-pro",
        "--local-dir",
        "/app/checkpoints/s2-pro",
        "--revision",
        FISH_S2_PRO_REVISION,
    ]
    assert len(FISH_S2_PRO_REVISION) == 40


def test_s2_profile_healthcheck_and_port_match_runtime(s2_profile):
    spec = get_runtime_template(s2_profile.default_runtime).render(
        s2_profile, DeploymentRequest(model=s2_profile.id)
    )
    assert s2_profile.healthcheck.port == spec.port
    assert s2_profile.healthcheck.path == spec.healthcheck_path


def test_s2_honours_runtime_options_and_nested_extra_cli_args(s2_profile):
    request = DeploymentRequest(
        model=s2_profile.id,
        custom_args={
            "runtime_args": {"extra_cli_args": "--api-key 'secret key'"},
            "engine_args": {"compile": True},
            "extra_env": {"HF_HOME": "/data/hf"},
        },
    )
    spec = get_runtime_template("fish-speech-s2").render(s2_profile, request)
    assert "--compile" in spec.run_args
    assert spec.run_args[-2:] == ["--api-key", "secret key"]
    assert spec.env_vars["HF_HOME"] == "/data/hf"


def test_legacy_fish_speech_template_still_v1():
    """Older Fish Speech models keep working on the v1.x runtime."""
    profile = ModelProfile(
        id="fish-speech-1.5",
        name="Fish Speech 1.5",
        workload_type=WorkloadType.AUDIO,
        default_runtime="fish-speech",
        hardware=HardwareRequirements(min_vram_gb=12.0),
        healthcheck=HealthcheckConfig(port=8080, path="/v1/health"),
    )
    spec = get_runtime_template("fish-speech").render(profile, DeploymentRequest(model=profile.id))
    assert isinstance(get_runtime_template("fish-speech"), FishSpeechTemplate)
    assert spec.docker_image == FISH_SPEECH_IMAGE


@pytest.mark.parametrize(
    "payload",
    [
        "; touch /tmp/pwned",
        "$(id)",
        "`id`",
        "org/repo && rm -rf /",
        "../../etc/passwd",
        "..",
        "payload 'with' \"quotes\" and spaces",
    ],
)
def test_s2_template_treats_artifact_as_data(payload):
    profile = ModelProfile(
        id="adv-s2",
        name="Adversarial",
        workload_type=WorkloadType.AUDIO,
        default_runtime="fish-speech-s2",
        artifact_id=payload,
        hardware=HardwareRequirements(min_vram_gb=24.0),
        healthcheck=HealthcheckConfig(port=8080),
    )
    spec = FishSpeechS2Template().render(profile, DeploymentRequest(model="adv-s2"))

    setup_tokens = shlex.split(spec.setup_commands[0])
    assert setup_tokens[2] == payload
    assert shlex.split(spec.run_command) == spec.run_args
    # The checkpoint directory never escapes /app/checkpoints.
    ckpt = spec.run_args[spec.run_args.index("--llama-checkpoint-path") + 1]
    assert ckpt.startswith("/app/checkpoints/")
    assert ".." not in ckpt.split("/")
    assert "--revision" not in setup_tokens  # revision pin applies to the S2 Pro artifact only
