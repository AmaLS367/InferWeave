"""Static GPU hardware catalog adapter with standard GPU specs and alias resolution."""

import re

from inferweave.models.routing import GpuSpec
from inferweave.ports.gpu_catalog import GpuCatalogPort


class StaticGpuCatalogAdapter(GpuCatalogPort):
    """Provides standard GPU hardware specifications, VRAM capacities, and alias matching."""

    def __init__(self, custom_specs: dict[str, GpuSpec] | None = None) -> None:
        self._specs: dict[str, GpuSpec] = custom_specs or self._build_default_specs()
        self._aliases: dict[str, str] = self._build_aliases()

    def get_gpu(self, name: str) -> GpuSpec | None:
        """Retrieves GPU specification by name with robust alias/fuzzy resolution."""
        if not name:
            return None

        # 1. Exact match in raw specs
        if name in self._specs:
            return self._specs[name]

        # 2. Normalized lookup
        norm = self._normalize_key(name)
        if norm in self._aliases:
            canon = self._aliases[norm]
            return self._specs.get(canon)

        # 3. Partial substring matching (e.g. "h100" in "h100-80gb" or "a100" in "a100-80gb")
        for alias, canon in self._aliases.items():
            if norm == alias or (len(norm) >= 3 and (norm in alias or alias in norm)):
                return self._specs.get(canon)

        return None

    def list_gpus(self) -> list[GpuSpec]:
        """Returns all unique GPU specifications."""
        seen: set[str] = set()
        result: list[GpuSpec] = []
        for spec in self._specs.values():
            if spec.name not in seen:
                seen.add(spec.name)
                result.append(spec)
        return result

    def find_sufficient_gpus(
        self, min_vram_gb: float, gpu_count: int = 1
    ) -> list[GpuSpec]:
        """Finds all GPU models that satisfy the minimum VRAM requirement, sorted by VRAM."""
        count = max(1, gpu_count)
        viable = [g for g in self.list_gpus() if g.total_vram(count) >= min_vram_gb]
        viable.sort(key=lambda g: g.vram_gb)
        return viable

    @staticmethod
    def _normalize_key(name: str) -> str:
        """Strips whitespace, hyphens, underscores and lowers text."""
        return re.sub(r"[\s\-_]+", "", name).lower()

    def _build_default_specs(self) -> dict[str, GpuSpec]:
        return {
            "T4": GpuSpec(
                name="T4", vram_gb=16.0, architecture="Turing", compute_capability=7.5
            ),
            "V100-16GB": GpuSpec(
                name="V100-16GB",
                vram_gb=16.0,
                architecture="Volta",
                compute_capability=7.0,
            ),
            "V100-32GB": GpuSpec(
                name="V100-32GB",
                vram_gb=32.0,
                architecture="Volta",
                compute_capability=7.0,
            ),
            "L4": GpuSpec(
                name="L4",
                vram_gb=24.0,
                architecture="Ada Lovelace",
                compute_capability=8.9,
            ),
            "A10G": GpuSpec(
                name="A10G", vram_gb=24.0, architecture="Ampere", compute_capability=8.6
            ),
            "A10": GpuSpec(
                name="A10", vram_gb=24.0, architecture="Ampere", compute_capability=8.6
            ),
            "RTX3090": GpuSpec(
                name="RTX3090",
                vram_gb=24.0,
                architecture="Ampere",
                compute_capability=8.6,
            ),
            "RTX4090": GpuSpec(
                name="RTX4090",
                vram_gb=24.0,
                architecture="Ada Lovelace",
                compute_capability=8.9,
            ),
            "L40S": GpuSpec(
                name="L40S",
                vram_gb=48.0,
                architecture="Ada Lovelace",
                compute_capability=8.9,
            ),
            "A100-40GB": GpuSpec(
                name="A100-40GB",
                vram_gb=40.0,
                architecture="Ampere",
                compute_capability=8.0,
            ),
            "A100-80GB": GpuSpec(
                name="A100-80GB",
                vram_gb=80.0,
                architecture="Ampere",
                compute_capability=8.0,
            ),
            "H100": GpuSpec(
                name="H100", vram_gb=80.0, architecture="Hopper", compute_capability=9.0
            ),
            "H200": GpuSpec(
                name="H200",
                vram_gb=141.0,
                architecture="Hopper",
                compute_capability=9.0,
            ),
            "B200": GpuSpec(
                name="B200",
                vram_gb=192.0,
                architecture="Blackwell",
                compute_capability=10.0,
            ),
        }

    def _build_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for canon in self._specs:
            aliases[self._normalize_key(canon)] = canon

        # Common short aliases
        extra = {
            "t4": "T4",
            "v100": "V100-16GB",
            "l4": "L4",
            "a10g": "A10G",
            "a10": "A10",
            "3090": "RTX3090",
            "rtx3090": "RTX3090",
            "4090": "RTX4090",
            "rtx4090": "RTX4090",
            "l40s": "L40S",
            "a10040gb": "A100-40GB",
            "a10040g": "A100-40GB",
            "a10040": "A100-40GB",
            "a10080gb": "A100-80GB",
            "a10080g": "A100-80GB",
            "a10080": "A100-80GB",
            "a100": "A100-80GB",
            "h100": "H100",
            "h200": "H200",
            "b200": "B200",
        }
        for k, v in extra.items():
            norm_k = self._normalize_key(k)
            if norm_k not in aliases:
                aliases[norm_k] = v

        return aliases
