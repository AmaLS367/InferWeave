"""Application service coordinating GPU hardware discovery, capacity checking, and pre-flight validation."""

import logging

from inferweave.adapters.catalog.static_gpu_catalog import StaticGpuCatalogAdapter
from inferweave.core.exceptions import UnknownGpuError
from inferweave.domain.hardware_validator import HardwareValidator
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import ModelProfile
from inferweave.models.routing import GpuSpec, VramCheckResult
from inferweave.ports.gpu_catalog import GpuCatalogPort

logger = logging.getLogger(__name__)


class HardwareValidationService:
    """Coordinates hardware validation ports, GPU catalogs, and domain validation rules."""

    def __init__(
        self,
        gpu_catalog: GpuCatalogPort | None = None,
        validator: HardwareValidator | None = None,
    ) -> None:
        self._gpu_catalog = gpu_catalog or StaticGpuCatalogAdapter()
        self._validator = validator or HardwareValidator()

    def get_gpu_spec(self, name: str) -> GpuSpec | None:
        """Looks up GPU hardware specification by name or alias."""
        return self._gpu_catalog.get_gpu(name)

    def list_known_gpus(self) -> list[GpuSpec]:
        """Lists all known GPU hardware specifications."""
        return self._gpu_catalog.list_gpus()

    def find_sufficient_gpus(
        self, min_vram_gb: float, gpu_count: int = 1
    ) -> list[GpuSpec]:
        """Finds all GPU models with sufficient VRAM for the requirement."""
        return self._gpu_catalog.find_sufficient_gpus(min_vram_gb, gpu_count)

    def check_compatibility(
        self,
        profile: ModelProfile,
        gpu_name: str,
        gpu_count: int = 1,
    ) -> VramCheckResult | None:
        """Checks VRAM compatibility for a specific GPU name without raising an exception."""
        spec = self._gpu_catalog.get_gpu(gpu_name)
        if not spec:
            return None
        return self._validator.check_vram_compatibility(
            required_vram_gb=profile.hardware.min_vram_gb,
            gpu_spec=spec,
            gpu_count=gpu_count,
            known_gpus=self._gpu_catalog.list_gpus(),
        )

    def validate_deployment_hardware(
        self,
        profile: ModelProfile,
        request: DeploymentRequest,
        strict: bool = False,
    ) -> GpuSpec | None:
        """Pre-flight check verifying requested GPU hardware satisfies model VRAM requirements.

        Raises:
            InsufficientVramError: If target GPU does not have enough VRAM.
            UnknownGpuError: If GPU model is unknown and strict mode is enabled.
        """
        if not request.gpu_type:
            return None

        gpu_spec = self._gpu_catalog.get_gpu(request.gpu_type)
        if gpu_spec is None:
            if strict:
                raise UnknownGpuError(request.gpu_type)
            logger.warning(
                "Requested GPU '%s' for model '%s' is not in standard GPU catalog; "
                "proceeding without offline VRAM validation.",
                request.gpu_type,
                profile.id,
            )
            return None

        gpu_count = request.num_gpus or profile.hardware.gpu_count
        self._validator.validate(
            profile=profile,
            gpu_spec=gpu_spec,
            gpu_count=gpu_count,
            known_gpus=self._gpu_catalog.list_gpus(),
        )
        return gpu_spec
