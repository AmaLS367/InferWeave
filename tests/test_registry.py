"""Registry lookups, built-in open LLM profiles, aliases and not-found diagnostics."""

import pytest

from inferweave import InferWeave, WorkloadType
from inferweave.core.exceptions import ModelNotFoundError
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import HardwareRequirements, ModelProfile
from inferweave.registry.base import ModelRegistry
from inferweave.runtimes.templates import get_runtime_template

OPEN_LLMS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]


@pytest.mark.parametrize("model_id", OPEN_LLMS)
def test_open_llms_are_vllm_profiles(model_id):
    profile = ModelRegistry().get(model_id)
    assert profile.workload_type == WorkloadType.LLM
    assert profile.default_runtime == "vllm"
    assert profile.hardware.min_vram_gb <= 24.0, "must fit an L4/A10G"
    spec = get_runtime_template("vllm").render(profile, DeploymentRequest(model=profile.id))
    assert spec.run_args[spec.run_args.index("--model") + 1] == profile.target_artifact


def test_llama_32_uses_ungated_artifact_but_keeps_meta_id():
    profile = ModelRegistry().get("meta-llama/Llama-3.2-3B-Instruct")
    assert profile.artifact_id == "unsloth/Llama-3.2-3B-Instruct"


def test_every_builtin_alias_points_to_a_registered_model():
    registry = ModelRegistry()
    ids = {m.id for m in registry.list_models()}
    assert registry.aliases()
    assert set(registry.aliases().values()) <= ids


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("flux", "black-forest-labs/FLUX.1-schnell"),
        ("wan2.1", "wan-video/wan-2.1"),
        ("llama-3.2-3b", "meta-llama/Llama-3.2-3B-Instruct"),
        ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
        ("qwen/qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("fish-s2-pro", "fish-s2-pro"),
    ],
)
def test_lookup_resolves_aliases_case_and_bare_names(name, expected):
    assert ModelRegistry().get(name).id == expected


def test_exact_id_wins_over_alias():
    registry = ModelRegistry()
    custom = ModelProfile(
        id="flux",
        name="Custom flux",
        workload_type=WorkloadType.IMAGE,
        default_runtime="flux-diffusers",
        hardware=HardwareRequirements(min_vram_gb=8.0),
    )
    registry.register(custom)
    assert registry.get("flux") is custom


def test_ambiguous_bare_name_is_not_guessed():
    registry = ModelRegistry()
    for org in ("a", "b"):
        registry.register(
            ModelProfile(
                id=f"{org}/shared-name",
                name=org,
                workload_type=WorkloadType.LLM,
                default_runtime="vllm",
                hardware=HardwareRequirements(min_vram_gb=8.0),
            )
        )
    with pytest.raises(ModelNotFoundError):
        registry.get("shared-name")


def test_register_alias_requires_registered_model():
    with pytest.raises(ModelNotFoundError):
        ModelRegistry().register_alias("ghost", "nobody/ghost")


def test_not_found_error_lists_models_and_suggests_close_matches():
    with pytest.raises(ModelNotFoundError) as info:
        ModelRegistry().get("Qwen/Qwen2.5-3B-Instruct")
    message = str(info.value)
    assert "Registered models:" in message
    assert "Did you mean: Qwen/Qwen2.5" in message
    assert "meta-llama/Llama-3.2-3B-Instruct" in info.value.available
    assert "weave.register_model(...)" in message


def test_not_found_error_without_candidates_keeps_legacy_message():
    message = str(ModelNotFoundError("x", reason="gone"))
    assert message == (
        "Model 'x' was not found in the InferWeave registry (gone). "
        "You can register it using weave.register_model(...)."
    )


def test_profile_engine_args_cap_context_and_request_overrides_them():
    profile = ModelRegistry().get("meta-llama/Llama-3.2-3B-Instruct")
    template = get_runtime_template("vllm")

    default = template.render(profile, DeploymentRequest(model=profile.id))
    i = default.run_args.index("--max-model-len")
    assert default.run_args[i + 1] == "16384"

    override = template.render(
        profile,
        DeploymentRequest(
            model=profile.id,
            custom_args={"engine_args": {"max_model_len": 4096}},
        ),
    )
    assert override.run_args.count("--max-model-len") == 1
    j = override.run_args.index("--max-model-len")
    assert override.run_args[j + 1] == "4096"


def test_profile_without_engine_args_renders_as_before():
    profile = ModelRegistry().get("meta-llama/Meta-Llama-3-8B-Instruct")
    spec = get_runtime_template("vllm").render(profile, DeploymentRequest(model=profile.id))
    assert "--max-model-len" not in spec.run_args


@pytest.mark.asyncio
async def test_deploy_through_alias_records_canonical_model():
    weave = InferWeave()
    deployment = await weave.deploy(model="flux", provider="modal", dry_run=True)
    assert deployment.model == "black-forest-labs/FLUX.1-schnell"
