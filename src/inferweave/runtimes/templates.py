"""Built-in runtime templates for Audio, LLM, Image, and Video workloads."""

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import ModelProfile
from inferweave.runtimes.base import RuntimeSpec, RuntimeTemplate


class VLLMTemplate(RuntimeTemplate):
    """Production runtime template for LLMs powered by vLLM."""

    @property
    def name(self) -> str:
        return "vllm"

    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        gpu_count = request.num_gpus or profile.hardware.gpu_count
        port = profile.healthcheck.port or 8000
        runtime_opts = request.options.runtime if request.options else None

        cmd_parts = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            profile.target_artifact,
            "--port",
            str(port),
            "--host",
            "0.0.0.0",
        ]

        # Resolve tensor parallel size
        tp_size = None
        if runtime_opts and "tensor_parallel_size" in runtime_opts.engine_args:
            tp_size = runtime_opts.engine_args.get("tensor_parallel_size")
        if tp_size is None:
            tp_size = gpu_count
        cmd_parts.extend(["--tensor-parallel-size", str(tp_size)])

        # Trust remote code flag
        trust_remote = True
        if runtime_opts and "trust_remote_code" in runtime_opts.engine_args:
            trust_remote = bool(runtime_opts.engine_args.get("trust_remote_code"))
        if trust_remote:
            cmd_parts.append("--trust-remote-code")

        # Append additional structured engine arguments and CLI flags
        if runtime_opts:
            handled = {"tensor_parallel_size", "trust_remote_code"}
            for key, val in runtime_opts.engine_args.items():
                if key in handled or val is None or val is False:
                    continue
                flag_name = key if key.startswith("-") else f"--{key.replace('_', '-')}"
                if val is True:
                    cmd_parts.append(flag_name)
                else:
                    cmd_parts.extend([flag_name, str(val)])
            cmd_parts.extend(runtime_opts.extra_cli_args)

        cmd = " ".join(cmd_parts)
        extra_env = runtime_opts.extra_env if runtime_opts else {}
        env = {**profile.default_env, **request.env, **extra_env}

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
        runtime_opts = request.options.runtime if request.options else None
        if runtime_opts:
            extra_cli = runtime_opts.to_cli_args()
            if extra_cli:
                cmd = f"{cmd} {' '.join(extra_cli)}"
            extra_env = runtime_opts.extra_env
        else:
            extra_env = {}

        env = {**profile.default_env, **request.env, **extra_env}

        return RuntimeSpec(
            name=self.name,
            docker_image="fishaudio/fish-speech:latest-cu121",
            setup_commands=[
                f"huggingface-cli download {profile.target_artifact} --local-dir checkpoints/{profile.id}"
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
        cmd = f"python3 -m inferweave.workers.flux --model {profile.target_artifact} --port {port}"
        runtime_opts = request.options.runtime if request.options else None
        if runtime_opts:
            extra_cli = runtime_opts.to_cli_args()
            if extra_cli:
                cmd = f"{cmd} {' '.join(extra_cli)}"
            extra_env = runtime_opts.extra_env
        else:
            extra_env = {}

        env = {**profile.default_env, **request.env, **extra_env}

        return RuntimeSpec(
            name=self.name,
            docker_image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
            setup_commands=[
                "pip install -U inferweave diffusers transformers accelerate sentencepiece protobuf fastapi uvicorn"
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
        cmd = f"python3 -m inferweave.workers.wan --model {profile.target_artifact} --port {port}"
        runtime_opts = request.options.runtime if request.options else None
        if runtime_opts:
            extra_cli = runtime_opts.to_cli_args()
            if extra_cli:
                cmd = f"{cmd} {' '.join(extra_cli)}"
            extra_env = runtime_opts.extra_env
        else:
            extra_env = {}

        env = {**profile.default_env, **request.env, **extra_env}

        return RuntimeSpec(
            name=self.name,
            docker_image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
            setup_commands=[
                "pip install -U inferweave diffusers transformers accelerate sentencepiece protobuf fastapi uvicorn"
            ],
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
