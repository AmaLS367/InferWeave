"""Domain service for validating hardware requirements, GPU capacity, and VRAM sizing."""

import logging

from inferweave.core.exceptions import InsufficientVramError
from inferweave.models.profile import ModelProfile
from inferweave.models.routing import GpuSpec, VramCheckResult

logger = logging.getLogger(__name__)


class HardwareValidator:
    """Validates whether target GPU hardware satisfies model hardware requirements."""

    def check_vram_compatibility(
        self,
        required_vram_gb: float,
        gpu_spec: GpuSpec,
        gpu_count: int | None = 1,
        known_gpus: list[GpuSpec] | None = None,
    ) -> VramCheckResult:
        """Evaluates whether the specified GPU and count provide sufficient VRAM."""
        count = max(1, gpu_count or 1)
        provided_vram = gpu_spec.total_vram(count)
        is_compatible = provided_vram >= required_vram_gb
        deficit = max(0.0, required_vram_gb - provided_vram)

        suggested: list[str] = []
        if not is_compatible and known_gpus:
            # Suggest GPUs that satisfy requirement with count=1 or count=count
            viable = [
                g.name
                for g in known_gpus
                if g.total_vram(count) >= required_vram_gb and g.name != gpu_spec.name
            ]
            suggested = viable[:4]

        return VramCheckResult(
            is_compatible=is_compatible,
            required_vram_gb=required_vram_gb,
            provided_vram_gb=provided_vram,
            deficit_gb=deficit,
            gpu_spec=gpu_spec,
            gpu_count=count,
            suggested_gpus=suggested,
        )

    def validate(
        self,
        profile: ModelProfile,
        gpu_spec: GpuSpec,
        gpu_count: int = 1,
        known_gpus: list[GpuSpec] | None = None,
    ) -> None:
        """Enforces VRAM constraint and raises InsufficientVramError on violation."""
        result = self.check_vram_compatibility(
            required_vram_gb=profile.hardware.min_vram_gb,
            gpu_spec=gpu_spec,
            gpu_count=gpu_count,
            known_gpus=known_gpus,
        )

        if not result.is_compatible:
            raise InsufficientVramError(
                model_id=profile.id,
                required_vram_gb=result.required_vram_gb,
                provided_vram_gb=result.provided_vram_gb,
                gpu_type=gpu_spec.name,
                gpu_count=gpu_count,
                suggested_gpus=result.suggested_gpus,
            )
