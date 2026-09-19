"""Model registry managing hardware profiles, default runtimes, and health specs."""

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
        self._register_defaults()

    def register(self, profile: ModelProfile) -> None:
        """Registers or overrides a model profile."""
        self._models[profile.id] = profile

    def get(self, model_id: str) -> ModelProfile:
        """Retrieves a model profile by identifier. Raises ModelNotFoundError if missing."""
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]

    def list_models(
        self, workload_type: WorkloadType | None = None
    ) -> list[ModelProfile]:
        """Lists registered models, optionally filtered by workload category."""
        if workload_type is None:
            return list(self._models.values())
        return [m for m in self._models.values() if m.workload_type == workload_type]

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
                default_runtime="fish-speech",
                hardware=HardwareRequirements(
                    min_vram_gb=16.0,
                    recommended_gpus=["A10G", "L4", "RTX4090", "A100"],
                    gpu_count=1,
                ),
                healthcheck=HealthcheckConfig(port=8080, path="/health"),
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
