"""Port interface for GPU hardware specifications catalog."""

from abc import ABC, abstractmethod

from inferweave.models.routing import GpuSpec


class GpuCatalogPort(ABC):
    """Abstract port for querying GPU hardware specifications and VRAM capacity."""

    @abstractmethod
    def get_gpu(self, name: str) -> GpuSpec | None:
        """Retrieves GPU specification by name (with alias/case-insensitive resolution)."""

    @abstractmethod
    def list_gpus(self) -> list[GpuSpec]:
        """Lists all known GPU hardware specifications."""

    @abstractmethod
    def find_sufficient_gpus(
        self, min_vram_gb: float, gpu_count: int = 1
    ) -> list[GpuSpec]:
        """Finds all GPU models that satisfy the minimum VRAM requirement."""
