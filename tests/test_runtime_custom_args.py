"""Unit tests verifying Runtime templates correctly incorporate custom_args and RuntimeOptions."""

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)
from inferweave.runtimes.templates import (
    FishSpeechTemplate,
    FluxDiffusersTemplate,
    VLLMTemplate,
    WanVideoTemplate,
)


def sample_profile(model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> ModelProfile:
    return ModelProfile(
        id=model_id,
        name="Sample Model",
        workload_type=WorkloadType.LLM,
        default_runtime="vllm",
        hardware=HardwareRequirements(min_vram_gb=16.0, gpu_count=1),
        healthcheck=HealthcheckConfig(port=8000),
    )



def test_vllm_template_renders_custom_engine_args():
    template = VLLMTemplate()
    profile = sample_profile()
    request = DeploymentRequest(
        model=profile.id,
        custom_args={
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
            "dtype": "bfloat16",
            "quantization": "awq",
            "enforce_eager": True,
            "extra_cli_args": ["--enable-chunked-prefill"],
            "extra_env": {"VLLM_LOGGING_LEVEL": "DEBUG"},
        },
    )

    spec = template.render(profile, request)

    assert "--max-model-len 4096" in spec.run_command
    assert "--gpu-memory-utilization 0.85" in spec.run_command
    assert "--dtype bfloat16" in spec.run_command
    assert "--quantization awq" in spec.run_command
    assert "--enforce-eager" in spec.run_command
    assert "--enable-chunked-prefill" in spec.run_command
    assert spec.env_vars.get("VLLM_LOGGING_LEVEL") == "DEBUG"


def test_vllm_template_overrides_tensor_parallel_size():
    template = VLLMTemplate()
    profile = sample_profile()
    request = DeploymentRequest(
        model=profile.id,
        num_gpus=1,
        custom_args={"tensor_parallel_size": 4},
    )

    spec = template.render(profile, request)
    assert "--tensor-parallel-size 4" in spec.run_command


def test_fish_speech_template_custom_args():
    template = FishSpeechTemplate()
    profile = sample_profile("fish-s2-pro")
    request = DeploymentRequest(
        model=profile.id,
        custom_args={"extra_cli_args": ["--compile", "--half"]},
    )

    spec = template.render(profile, request)
    assert "--compile" in spec.run_command
    assert "--half" in spec.run_command


def test_flux_and_wan_templates_custom_args():
    flux = FluxDiffusersTemplate()
    wan = WanVideoTemplate()
    profile = sample_profile("flux-schnell")
    request = DeploymentRequest(
        model=profile.id,
        custom_args={"extra_cli_args": ["--quantize-4bit"]},
    )

    flux_spec = flux.render(profile, request)
    wan_spec = wan.render(profile, request)

    assert "--quantize-4bit" in flux_spec.run_command
    assert "--quantize-4bit" in wan_spec.run_command
