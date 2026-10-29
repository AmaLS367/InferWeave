"""Model registry managing hardware profiles, default runtimes, and health specs."""

from typing import Any

from inferweave.core.exceptions import ModelNotFoundError
from inferweave.models.enums import WorkloadType
from inferweave.models.profile import (
    HardwareRequirements,
    HealthcheckConfig,
    ModelProfile,
)


class ModelRegistry:
    """Registry maintaining AI model profiles and computational requirements."""

    def __init__(self) -> None:
        self._models: dict[str, ModelProfile] = {}
        self._aliases: dict[str, str] = {}
        self._register_defaults()

    def register(self, profile: ModelProfile) -> None:
        """Registers or overrides a model profile."""
        self._models[profile.id] = profile

    def register_alias(self, alias: str, model_id: str) -> None:
        """Maps a short, case-insensitive ``alias`` to a registered model id."""
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        self._aliases[alias.casefold()] = model_id

    def aliases(self) -> dict[str, str]:
        """Returns a copy of the alias to model id mapping."""
        return dict(self._aliases)

    def get(self, model_id: str) -> ModelProfile:
        """Retrieves a model profile by identifier. Raises ModelNotFoundError if missing.

        Lookup order: exact id, registered alias, case-insensitive id, then the repository-less
        name (``Qwen2.5-1.5B-Instruct`` for ``Qwen/Qwen2.5-1.5B-Instruct``) when it is unique.
        """
        resolved = self._resolve(model_id)
        if resolved is None:
            raise ModelNotFoundError(model_id, available=sorted(self._models))
        return self._models[resolved]

    def list_models(
        self, workload_type: WorkloadType | None = None
    ) -> list[ModelProfile]:
        """Lists registered models, optionally filtered by workload category."""
        if workload_type is None:
            return list(self._models.values())
        return [m for m in self._models.values() if m.workload_type == workload_type]

    def _resolve(self, model_id: str) -> str | None:
        if model_id in self._models:
            return model_id
        key = model_id.casefold()
        if key in self._aliases:
            return self._aliases[key]
        by_id = [mid for mid in self._models if mid.casefold() == key]
        if len(by_id) == 1:
            return by_id[0]
        by_name = [mid for mid in self._models if mid.rsplit("/", 1)[-1].casefold() == key]
        if len(by_name) == 1:
            return by_name[0]
        return None

    def _register_llm(
        self,
        model_id: str,
        name: str,
        min_vram_gb: float,
        *,
        artifact_id: str | None = None,
        recommended_gpus: list[str] | None = None,
        max_model_len: int | None = None,
        aliases: tuple[str, ...] = (),
        notes: str | None = None,
    ) -> None:
        """Registers an open-weight vLLM chat model and its aliases."""
        metadata: dict[str, Any] = {}
        if max_model_len is not None:
            # Keeps the KV cache of long-context models within small-GPU memory.
            metadata["vllm_engine_args"] = {"max_model_len": max_model_len}
        if notes:
            metadata["notes"] = notes
        self.register(
            ModelProfile(
                id=model_id,
                name=name,
                source="huggingface",
                artifact_id=artifact_id or model_id,
                workload_type=WorkloadType.LLM,
                default_runtime="vllm",
                hardware=HardwareRequirements(
                    min_vram_gb=min_vram_gb,
                    recommended_gpus=recommended_gpus or ["T4", "L4", "A10G", "A100"],
                    gpu_count=1,
                ),
                healthcheck=HealthcheckConfig(port=8000, path="/health"),
                metadata=metadata,
            )
        )
        for alias in aliases:
            self.register_alias(alias, model_id)

    def _register_defaults(self) -> None:
        """Populates built-in profiles for prominent open-weight models."""
        # Audio / TTS / Voice cloning
        self.register(
            ModelProfile(
                id="fish-s2-pro",
                name="Fish Speech S2 Pro",
                source="huggingface",
                artifact_id="fishaudio/s2-pro",
                workload_type=WorkloadType.AUDIO,
                default_runtime="fish-speech-s2",
                hardware=HardwareRequirements(
                    min_vram_gb=24.0,
                    recommended_gpus=["L4", "RTX4090", "A100", "H100"],
                    gpu_count=1,
                ),
                healthcheck=HealthcheckConfig(port=8080, path="/v1/health"),
            )
        )

        # LLM
        self.register(
            ModelProfile(
                id="meta-llama/Meta-Llama-3-8B-Instruct",
                name="Llama 3 8B Instruct",
                source="huggingface",
                artifact_id="meta-llama/Meta-Llama-3-8B-Instruct",
                workload_type=WorkloadType.LLM,
                default_runtime="vllm",
                hardware=HardwareRequirements(
                    min_vram_gb=16.0,
                    recommended_gpus=["A10G", "L4", "A100"],
                    gpu_count=1,
                ),
                healthcheck=HealthcheckConfig(port=8000, path="/health"),
            )
        )
        self.register_alias("llama-3-8b", "meta-llama/Meta-Llama-3-8B-Instruct")

        # Open, ungated chat models served by vLLM. Llama 3.2 uses the ``unsloth`` re-upload
        # because the ``meta-llama`` repositories are gated and need a Hugging Face token.
        self._register_llm(
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen2.5 0.5B Instruct",
            6.0,
            aliases=("qwen2.5-0.5b",),
        )
        self._register_llm(
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen2.5 1.5B Instruct",
            8.0,
            aliases=("qwen2.5-1.5b",),
        )
        self._register_llm(
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen2.5 7B Instruct",
            20.0,
            recommended_gpus=["L4", "A10G", "A100"],
            aliases=("qwen2.5-7b", "qwen2.5"),
        )
        self._register_llm(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen2.5 Coder 7B Instruct",
            20.0,
            recommended_gpus=["L4", "A10G", "A100"],
            aliases=("qwen2.5-coder-7b", "qwen2.5-coder"),
        )
        self._register_llm(
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Mistral 7B Instruct v0.3",
            20.0,
            recommended_gpus=["L4", "A10G", "A100"],
            aliases=("mistral-7b",),
        )
        self._register_llm(
            "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "SmolLM2 1.7B Instruct",
            8.0,
            aliases=("smollm2-1.7b", "smollm2"),
        )
        self._register_llm(
            "microsoft/Phi-3.5-mini-instruct",
            "Phi-3.5 Mini Instruct",
            16.0,
            max_model_len=16384,
            aliases=("phi-3.5-mini", "phi-3.5"),
        )
        self._register_llm(
            "meta-llama/Llama-3.2-1B-Instruct",
            "Llama 3.2 1B Instruct",
            8.0,
            artifact_id="unsloth/Llama-3.2-1B-Instruct",
            max_model_len=16384,
            aliases=("llama-3.2-1b",),
            notes="Ungated unsloth re-upload; Llama 3.2 Community License applies.",
        )
        self._register_llm(
            "meta-llama/Llama-3.2-3B-Instruct",
            "Llama 3.2 3B Instruct",
            12.0,
            artifact_id="unsloth/Llama-3.2-3B-Instruct",
            max_model_len=16384,
            aliases=("llama-3.2-3b",),
            notes="Ungated unsloth re-upload; Llama 3.2 Community License applies.",
        )

        # Image
        self.register(
            ModelProfile(
                id="black-forest-labs/FLUX.1-schnell",
                name="FLUX.1 Schnell",
                source="huggingface",
                artifact_id="black-forest-labs/FLUX.1-schnell",
                workload_type=WorkloadType.IMAGE,
                default_runtime="flux-diffusers",
                hardware=HardwareRequirements(
                    min_vram_gb=24.0,
                    recommended_gpus=["A100", "L40S", "RTX4090"],
                    gpu_count=1,
                ),
                healthcheck=HealthcheckConfig(port=8000, path="/health"),
            )
        )
        self.register_alias("flux", "black-forest-labs/FLUX.1-schnell")

        # Video
        self.register(
            ModelProfile(
                id="wan-video/wan-2.1",
                name="WAN 2.1 Video Generator",
                source="huggingface",
                artifact_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                workload_type=WorkloadType.VIDEO,
                default_runtime="wan-video",
                hardware=HardwareRequirements(
                    min_vram_gb=24.0,
                    recommended_gpus=["A100", "H100"],
                    gpu_count=1,
                ),
                healthcheck=HealthcheckConfig(port=8000, path="/health"),
            )
        )
        self.register_alias("wan2.1", "wan-video/wan-2.1")
        self.register_alias("wan", "wan-video/wan-2.1")
