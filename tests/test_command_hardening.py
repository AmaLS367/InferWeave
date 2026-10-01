"""Adversarial regression tests for runtime shell command construction and execution.

Validates that shell metacharacters (; $(...) && | quotes spaces) in model artifacts,
identifiers, structured engine arguments, and extra_cli_args cannot create unexpected
command injection or break execution semantics.
"""

import shlex
from unittest.mock import patch

import pytest

from inferweave.domain.options import DeploymentOptions
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.providers.modal_provider import ModalProvider
from inferweave.runtimes.base import RuntimeSpec
from inferweave.runtimes.templates import (
    FishSpeechTemplate,
    FluxDiffusersTemplate,
    VLLMTemplate,
    WanVideoTemplate,
)


def _make_profile(
    model_id: str,
    target_artifact: str,
    workload: WorkloadType = WorkloadType.LLM,
    runtime: str = "vllm",
) -> ModelProfile:
    return ModelProfile(
        id=model_id,
        name="Adversarial Model Probe",
        workload_type=workload,
        default_runtime=runtime,
        artifact_id=target_artifact,
        hardware=HardwareRequirements(min_vram_gb=16, gpu_count=1),
        healthcheck=HealthcheckConfig(port=8000),
    )


ADVERSARIAL_PAYLOADS = [
    "; touch /tmp/pwned",
    "$(id)",
    "`touch /tmp/pwned`",
    "&& rm -rf /tmp/important",
    "| cat /etc/passwd",
    "payload 'with' \"mixed\" quotes and spaces",
    "https://example.com/weights.bin?token=secret&format=safetensors",
    "/var/data/models/path with spaces/checkpoint",
    "--option=key:value;danger",
]


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_vllm_template_neutralizes_adversarial_artifact_id(payload):
    """Artifact IDs containing shell syntax must be safely treated as argument data."""
    template = VLLMTemplate()
    profile = _make_profile("adv-model", payload)
    request = DeploymentRequest(model=profile.id)

    spec = template.render(profile, request)

    # 1. Structured run_args must contain the literal payload without modification
    assert spec.run_args[spec.run_args.index("--model") + 1] == payload

    # 2. Parsing the shell run_command with shlex.split must recover the exact same tokens
    recovered = shlex.split(spec.run_command)
    assert recovered == spec.run_args
    assert recovered[recovered.index("--model") + 1] == payload


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_vllm_template_neutralizes_adversarial_engine_args(payload):
    """Engine arguments with metacharacters must be safely passed as discrete argv tokens."""
    template = VLLMTemplate()
    profile = _make_profile("adv-model", "meta-llama/Meta-Llama-3-8B-Instruct")
    request = DeploymentRequest(
        model=profile.id,
        custom_args={
            "chat_template": payload,
            "extra_cli_args": [payload, f"--custom={payload}"],
        },
    )

    spec = template.render(profile, request)

    # In argv, each item must be pure data
    assert payload in spec.run_args
    assert f"--custom={payload}" in spec.run_args

    recovered = shlex.split(spec.run_command)
    assert recovered == spec.run_args
    assert payload in recovered


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_fish_speech_setup_commands_and_run_command_safe(payload):
    """FishSpeech template setup_commands and launch command must quote artifact and paths."""
    template = FishSpeechTemplate()
    profile = _make_profile("adv-voice-model", payload, WorkloadType.AUDIO, "fish-speech")
    request = DeploymentRequest(
        model=profile.id,
        custom_args={"extra_cli_args": [f"--tag={payload}"]},
    )

    spec = template.render(profile, request)

    # Verify setup command: huggingface-cli download
    assert len(spec.setup_commands) == 1
    setup_tokens = shlex.split(spec.setup_commands[0])
    assert setup_tokens[0] == "huggingface-cli"
    assert setup_tokens[1] == "download"
    assert setup_tokens[2] == payload

    # Verify run_args and run_command
    assert shlex.split(spec.run_command) == spec.run_args
    assert f"--tag={payload}" in spec.run_args


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_flux_and_wan_workers_treat_payloads_as_data(payload):
    """FLUX and WAN templates must treat model paths and CLI args as structured data."""
    for template_cls, workload in ((FluxDiffusersTemplate, WorkloadType.IMAGE), (WanVideoTemplate, WorkloadType.VIDEO)):
        template = template_cls()
        profile = _make_profile("adv-gen", payload, workload, template.name)
        request = DeploymentRequest(
            model=profile.id,
            custom_args={"extra_cli_args": [payload]},
        )

        spec = template.render(profile, request)
        assert spec.run_args[spec.run_args.index("--model") + 1] == payload
        assert payload in spec.run_args
        assert shlex.split(spec.run_command) == spec.run_args


def test_extra_cli_args_parsing_from_string_or_list():
    """DeploymentOptions.from_custom_args safely parses space-separated strings or lists."""
    # From string with spaces and quotes
    opts_str = DeploymentOptions.from_custom_args({
        "extra_cli_args": '--foo "val with space" --bar=123'
    })
    assert opts_str.runtime.extra_cli_args == ["--foo", "val with space", "--bar=123"]

    # From list
    opts_list = DeploymentOptions.from_custom_args({
        "extra_cli_args": ["--foo", "val with space", "--bar=123"]
    })
    assert opts_list.runtime.extra_cli_args == ["--foo", "val with space", "--bar=123"]


@pytest.mark.asyncio
async def test_modal_provider_executes_with_shell_false():
    """ModalProvider must invoke subprocess.Popen with shell=False and structured argv."""
    provider = ModalProvider()
    profile = _make_profile("safe-model", "test/model")
    runtime = RuntimeSpec(
        name="test",
        docker_image="python:3.11-slim",
        run_command="python3 -m app --param 'with space'",
        run_args=["python3", "-m", "app", "--param", "with space"],
        port=8000,
    )
    request = DeploymentRequest(model=profile.id, provider="modal", dry_run=False)

    captured_serve_fn = None

    def fake_web_server(**kwargs):
        def decorator(fn):
            nonlocal captured_serve_fn
            captured_serve_fn = fn
            return fn
        return decorator

    with (
        patch("modal.App.deploy", return_value=None),
        patch("modal.Function.get_web_url", return_value="https://test.modal.run"),
        patch("modal.web_server", side_effect=fake_web_server),
        patch("subprocess.Popen") as mock_popen,
    ):
        await provider.deploy(request, profile, runtime)
        assert captured_serve_fn is not None

        # Call the captured serve function to verify subprocess.Popen execution
        captured_serve_fn()
        mock_popen.assert_called_once()
        cmd_arg, kwargs = mock_popen.call_args
        assert cmd_arg[0] == ["python3", "-m", "app", "--param", "with space"]
        assert kwargs.get("shell") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
async def test_skypilot_run_script_is_built_only_from_quoted_run_args(payload):
    """SkyPilot's Task.run is a shell string: it must be shlex.join(run_args), never run_command.

    A run_command that diverges from run_args (e.g. carrying an injected fragment) must be
    ignored, and metacharacters inside structured args must stay quoted literal tokens.
    """
    from unittest.mock import MagicMock

    from inferweave.providers.skypilot import SkyPilotProvider

    run_args = ["python3", "-m", "server", "--model", payload, f"--tag={payload}"]
    runtime = RuntimeSpec(
        name="probe",
        docker_image="python:3.11-slim",
        run_command="python3 -m server && touch /tmp/pwned",  # must NOT reach SkyPilot
        run_args=run_args,
        port=8000,
    )
    profile = _make_profile("safe-model", "test/model")
    provider = SkyPilotProvider(cloud_name="runpod")
    mock_sky = MagicMock()
    mock_sky.launch = MagicMock(return_value=(1, None))
    mock_sky.endpoints = MagicMock(return_value={8000: "http://1.2.3.4:8000"})
    mock_sky.clouds.CLOUD_REGISTRY.from_str.return_value = MagicMock()

    with (
        patch.object(provider, "_ensure_supported_platform", return_value=None),
        patch.object(provider, "_get_sky_module", return_value=mock_sky),
    ):
        await provider.deploy(
            DeploymentRequest(model=profile.id, provider="runpod"), profile, runtime
        )

    _, task_kwargs = mock_sky.Task.call_args
    run_script = task_kwargs["run"]
    assert run_script == shlex.join(run_args)
    assert shlex.split(run_script) == run_args


@pytest.mark.parametrize("payload", ADVERSARIAL_PAYLOADS)
def test_builtin_templates_run_command_is_exactly_joined_run_args(payload):
    """For every built-in template, the shell form is derived solely from the argv form."""
    from inferweave.runtimes.templates import _BUILTIN_TEMPLATES

    for name, template in _BUILTIN_TEMPLATES.items():
        profile = _make_profile("adv", payload, WorkloadType.LLM, name)
        request = DeploymentRequest(
            model="adv",
            custom_args={"extra_cli_args": [payload], "runtime_args": {"extra_cli_args": f"--n {shlex.quote(payload)}"}},
        )
        spec = template.render(profile, request)
        assert spec.run_command == shlex.join(spec.run_args), name
        assert payload in spec.run_args, name
