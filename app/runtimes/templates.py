"""Built-in runtime templates for Audio, LLM, Image, and Video workloads."""

from app.models.deployment import DeploymentRequest
from app.models.profile import ModelProfile
from app.runtimes.base import RuntimeSpec, RuntimeTemplate


class VLLMTemplate(RuntimeTemplate):
    """Production runtime template for LLMs powered by vLLM."""

    @property
    def name(self) -> str:
        return "vllm"

    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        gpu_count = request.num_gpus or profile.hardware.gpu_count
        port = profile.healthcheck.port or 8000

        cmd = (
            f"python3 -m vllm.entrypoints.openai.api_server "
            f"--model {profile.id} "
            f"--port {port} "
            f"--host 0.0.0.0 "
            f"--tensor-parallel-size {gpu_count} "
            f"--trust-remote-code"
        )

        env = {**profile.default_env, **request.env}

        return RuntimeSpec(
            name=self.name,
            docker_image="vllm/vllm-openai:latest",
            run_command=cmd,
            port=port,
            env_vars=env,
            healthcheck_path="/health",
        )


class FishSpeechTemplate(RuntimeTemplate):
    """Runtime template for Fish Audio / Fish Speech TTS and voice cloning models."""

    @property
    def name(self) -> str:
        return "fish-speech"

    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        port = profile.healthcheck.port or 8080
        cmd = f"python3 -m tools.api_server --listen 0.0.0.0:{port} --llama-checkpoint-path checkpoints/{profile.id}"
        env = {**profile.default_env, **request.env}

        return RuntimeSpec(
            name=self.name,
            docker_image="fishaudio/fish-speech:latest-cu121",
            setup_commands=[
                f"huggingface-cli download {profile.id} --local-dir checkpoints/{profile.id}"
            ],
            run_command=cmd,
            port=port,
            env_vars=env,
            healthcheck_path="/health",
        )


class FluxDiffusersTemplate(RuntimeTemplate):
    """Runtime template for FLUX image synthesis via Diffusers."""

    @property
    def name(self) -> str:
        return "flux-diffusers"

    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        port = profile.healthcheck.port or 8000
        cmd = f"python3 -m inferweave_worker.flux --model {profile.id} --port {port}"
        env = {**profile.default_env, **request.env}

        return RuntimeSpec(
            name=self.name,
            docker_image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
            setup_commands=[
                "pip install -U diffusers transformers accelerate sentencepiece protobuf fastapi uvicorn"
            ],
            run_command=cmd,
            port=port,
            env_vars=env,
            healthcheck_path="/health",
        )


class WanVideoTemplate(RuntimeTemplate):
    """Runtime template for WAN open-weights video generation."""

    @property
    def name(self) -> str:
        return "wan-video"

    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        port = profile.healthcheck.port or 8000
        cmd = f"python3 -m wan_server --model {profile.id} --port {port}"
        env = {**profile.default_env, **request.env}

        return RuntimeSpec(
            name=self.name,
            docker_image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
            run_command=cmd,
            port=port,
            env_vars=env,
            healthcheck_path="/health",
        )


_BUILTIN_TEMPLATES: dict[str, RuntimeTemplate] = {
    "vllm": VLLMTemplate(),
    "fish-speech": FishSpeechTemplate(),
    "flux-diffusers": FluxDiffusersTemplate(),
    "wan-video": WanVideoTemplate(),
}


def get_runtime_template(name: str) -> RuntimeTemplate:
    """Retrieves a runtime template by identifier."""
    if name not in _BUILTIN_TEMPLATES:
        raise ValueError(
            f"Runtime template '{name}' not found. Available: {list(_BUILTIN_TEMPLATES.keys())}"
        )
    return _BUILTIN_TEMPLATES[name]
