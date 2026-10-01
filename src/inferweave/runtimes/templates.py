"""Built-in runtime templates for Audio, LLM, Image, and Video workloads.

Built-in production templates use explicit container version tags and exact dependency pins
from ``inferweave.runtimes.manifest``. Custom user-defined templates can still choose their own
arbitrary container images or mutable versions if desired.
"""

import re
import shlex

from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import ModelProfile
from inferweave.runtimes.base import RuntimeSpec, RuntimeTemplate
from inferweave.runtimes.manifest import (
    ACCELERATE_SPEC,
    DIFFUSERS_SPEC,
    DIFFUSION_TRANSITIVE_PINS,
    FASTAPI_SPEC,
    FISH_S2_PRO_ARTIFACT,
    FISH_S2_PRO_REVISION,
    FISH_SPEECH_IMAGE,
    FISH_SPEECH_S2_IMAGE,
    IMAGEIO_FFMPEG_SPEC,
    IMAGEIO_SPEC,
    PROTOBUF_SPEC,
    PYTORCH_IMAGE,
    SENTENCEPIECE_SPEC,
    TRANSFORMERS_SPEC,
    UVICORN_SPEC,
    VLLM_IMAGE,
)


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

        cmd = shlex.join(cmd_parts)
        extra_env = runtime_opts.extra_env if runtime_opts else {}
        env = {**profile.default_env, **request.env, **extra_env}

        return RuntimeSpec(
            name=self.name,
            docker_image=VLLM_IMAGE,
            run_command=cmd,
            run_args=cmd_parts,
            port=port,
            env_vars=env,
            healthcheck_path="/health",
        )


class FishSpeechTemplate(RuntimeTemplate):
    """Runtime template for legacy Fish Speech v1.x models. S2 models use ``FishSpeechS2Template``."""

    @property
    def name(self) -> str:
        return "fish-speech"

    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        port = profile.healthcheck.port or 8080
        cmd_parts = [
            "python3",
            "-m",
            "tools.api_server",
            "--listen",
            f"0.0.0.0:{port}",
            "--llama-checkpoint-path",
            f"checkpoints/{profile.id}",
            "--decoder-checkpoint-path",
            f"checkpoints/{profile.id}/codec.pth",
        ]
        runtime_opts = request.options.runtime if request.options else None
        if runtime_opts:
            cmd_parts.extend(runtime_opts.to_cli_args())
            extra_env = runtime_opts.extra_env
        else:
            extra_env = {}

        cmd = shlex.join(cmd_parts)
        env = {**profile.default_env, **request.env, **extra_env}

        setup_cmd = shlex.join([
            "huggingface-cli",
            "download",
            profile.target_artifact,
            "--local-dir",
            f"checkpoints/{profile.id}",
        ])

        return RuntimeSpec(
            name=self.name,
            docker_image=FISH_SPEECH_IMAGE,
            setup_commands=[setup_cmd],
            run_command=cmd,
            run_args=cmd_parts,
            port=port,
            env_vars=env,
            healthcheck_path="/v1/health",
        )


class FishSpeechS2Template(RuntimeTemplate):
    """Runtime template for Fish Audio S2 series models (e.g. ``fishaudio/s2-pro``).

    S2 needs the Fish Speech v2 codebase (Dual-AR model, ModifiedDAC codec ``codec.pth``,
    ``modded_dac_vq`` decoder config), which the v1.x image used by ``FishSpeechTemplate``
    does not contain. Follows upstream: weights downloaded with ``hf download`` into
    ``checkpoints/<name>``, server started with ``tools/api_server.py``, ``/v1/health``,
    default port 8080. The official image keeps its uv environment in ``/app/.venv`` and
    expects checkpoints under ``/app/checkpoints``; its entrypoint is bypassed here so the
    structured argv is the only thing executed.
    """

    APP_DIR = "/app"

    @property
    def name(self) -> str:
        return "fish-speech-s2"

    @staticmethod
    def _checkpoint_name(profile: ModelProfile) -> str:
        base = profile.target_artifact.rsplit("/", 1)[-1]
        base = re.sub(r"[^A-Za-z0-9._-]", "-", base)
        return base if base.strip(".") else "s2-pro"

    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        port = profile.healthcheck.port or 8080
        checkpoint_dir = f"{self.APP_DIR}/checkpoints/{self._checkpoint_name(profile)}"
        cmd_parts = [
            f"{self.APP_DIR}/.venv/bin/python",
            f"{self.APP_DIR}/tools/api_server.py",
            "--listen",
            f"0.0.0.0:{port}",
            "--llama-checkpoint-path",
            checkpoint_dir,
            "--decoder-checkpoint-path",
            f"{checkpoint_dir}/codec.pth",
            "--decoder-config-name",
            "modded_dac_vq",
        ]
        runtime_opts = request.options.runtime if request.options else None
        if runtime_opts:
            cmd_parts.extend(runtime_opts.to_cli_args())
            extra_env = runtime_opts.extra_env
        else:
            extra_env = {}

        env = {**profile.default_env, **request.env, **extra_env}

        download = [
            f"{self.APP_DIR}/.venv/bin/hf",
            "download",
            profile.target_artifact,
            "--local-dir",
            checkpoint_dir,
        ]
        if profile.target_artifact == FISH_S2_PRO_ARTIFACT:
            download.extend(["--revision", FISH_S2_PRO_REVISION])

        return RuntimeSpec(
            name=self.name,
            docker_image=FISH_SPEECH_S2_IMAGE,
            setup_commands=[shlex.join(download)],
            run_command=shlex.join(cmd_parts),
            run_args=cmd_parts,
            port=port,
            env_vars=env,
            healthcheck_path="/v1/health",
        )


class FluxDiffusersTemplate(RuntimeTemplate):
    """Runtime template for FLUX image synthesis via Diffusers."""

    @property
    def name(self) -> str:
        return "flux-diffusers"

    def render(self, profile: ModelProfile, request: DeploymentRequest) -> RuntimeSpec:
        port = profile.healthcheck.port or 8000
        cmd_parts = [
            "python3",
            "-m",
            "inferweave.workers.flux",
            "--model",
            profile.target_artifact,
            "--port",
            str(port),
        ]
        runtime_opts = request.options.runtime if request.options else None
        if runtime_opts:
            cmd_parts.extend(runtime_opts.to_cli_args())
            extra_env = runtime_opts.extra_env
        else:
            extra_env = {}

        cmd = shlex.join(cmd_parts)
        env = {**profile.default_env, **request.env, **extra_env}

        setup_pip = shlex.join([
            "pip",
            "install",
            DIFFUSERS_SPEC,
            TRANSFORMERS_SPEC,
            ACCELERATE_SPEC,
            SENTENCEPIECE_SPEC,
            PROTOBUF_SPEC,
            FASTAPI_SPEC,
            UVICORN_SPEC,
            *DIFFUSION_TRANSITIVE_PINS,
        ])

        return RuntimeSpec(
            name=self.name,
            docker_image=PYTORCH_IMAGE,
            setup_commands=[setup_pip],
            run_command=cmd,
            run_args=cmd_parts,
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
        cmd_parts = [
            "python3",
            "-m",
            "inferweave.workers.wan",
            "--model",
            profile.target_artifact,
            "--port",
            str(port),
        ]
        runtime_opts = request.options.runtime if request.options else None
        if runtime_opts:
            cmd_parts.extend(runtime_opts.to_cli_args())
            extra_env = runtime_opts.extra_env
        else:
            extra_env = {}

        cmd = shlex.join(cmd_parts)
        env = {**profile.default_env, **request.env, **extra_env}

        setup_pip = shlex.join([
            "pip",
            "install",
            DIFFUSERS_SPEC,
            TRANSFORMERS_SPEC,
            ACCELERATE_SPEC,
            SENTENCEPIECE_SPEC,
            PROTOBUF_SPEC,
            FASTAPI_SPEC,
            UVICORN_SPEC,
            IMAGEIO_SPEC,
            IMAGEIO_FFMPEG_SPEC,
            *DIFFUSION_TRANSITIVE_PINS,
        ])

        return RuntimeSpec(
            name=self.name,
            docker_image=PYTORCH_IMAGE,
            setup_commands=[setup_pip],
            run_command=cmd,
            run_args=cmd_parts,
            port=port,
            env_vars=env,
            healthcheck_path="/health",
        )


_BUILTIN_TEMPLATES: dict[str, RuntimeTemplate] = {
    "vllm": VLLMTemplate(),
    "fish-speech": FishSpeechTemplate(),
    "fish-speech-s2": FishSpeechS2Template(),
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
