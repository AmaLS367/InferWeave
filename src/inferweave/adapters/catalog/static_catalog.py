"""Static catalog adapter providing built-in hardware specs, pricing, and latency defaults."""

from inferweave.models.routing import GpuSpec, InstanceOffer
from inferweave.ports.catalog import ProviderCatalogPort


class StaticCatalogAdapter(ProviderCatalogPort):
    """Provides curated, offline-compatible GPU cloud instance offerings and pricing.

    Ensures robust, instant (<1ms) routing without requiring network calls or cloud credentials.
    """

    def __init__(self, custom_offers: list[InstanceOffer] | None = None) -> None:
        self._offers: list[InstanceOffer] = custom_offers if custom_offers is not None else self._build_default_offers()

    async def get_offers(self, provider_name: str | None = None) -> list[InstanceOffer]:
        if provider_name is None:
            return list(self._offers)
        p_name = provider_name.lower()
        return [o for o in self._offers if o.provider.lower() == p_name]

    def _build_default_offers(self) -> list[InstanceOffer]:
        gpu_specs = {
            "T4": GpuSpec(name="T4", vram_gb=16.0, architecture="Turing", compute_capability=7.5),
            "L4": GpuSpec(name="L4", vram_gb=24.0, architecture="Ada Lovelace", compute_capability=8.9),
            "A10G": GpuSpec(name="A10G", vram_gb=24.0, architecture="Ampere", compute_capability=8.6),
            "A10": GpuSpec(name="A10", vram_gb=24.0, architecture="Ampere", compute_capability=8.6),
            "RTX3090": GpuSpec(name="RTX3090", vram_gb=24.0, architecture="Ampere", compute_capability=8.6),
            "RTX4090": GpuSpec(name="RTX4090", vram_gb=24.0, architecture="Ada Lovelace", compute_capability=8.9),
            "L40S": GpuSpec(name="L40S", vram_gb=48.0, architecture="Ada Lovelace", compute_capability=8.9),
            "A100-40GB": GpuSpec(name="A100-40GB", vram_gb=40.0, architecture="Ampere", compute_capability=8.0),
            "A100": GpuSpec(name="A100", vram_gb=80.0, architecture="Ampere", compute_capability=8.0),
            "A100-80GB": GpuSpec(name="A100-80GB", vram_gb=80.0, architecture="Ampere", compute_capability=8.0),
            "H100": GpuSpec(name="H100", vram_gb=80.0, architecture="Hopper", compute_capability=9.0),
        }

        offers: list[InstanceOffer] = [
            # --- Modal (Serverless, ultra-fast cold start, per-second billing) ---
            InstanceOffer(
                provider="modal",
                instance_type="modal.t4",
                gpu_spec=gpu_specs["T4"],
                price_per_hour=0.59,
                estimated_cold_start_sec=6.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="modal",
                instance_type="modal.l4",
                gpu_spec=gpu_specs["L4"],
                price_per_hour=0.80,
                estimated_cold_start_sec=7.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="modal",
                instance_type="modal.a10g",
                gpu_spec=gpu_specs["A10G"],
                price_per_hour=1.10,
                estimated_cold_start_sec=8.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="modal",
                instance_type="modal.a100-40gb",
                gpu_spec=gpu_specs["A100-40GB"],
                price_per_hour=2.10,
                estimated_cold_start_sec=12.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="modal",
                instance_type="modal.a100-80gb",
                gpu_spec=gpu_specs["A100-80GB"],
                price_per_hour=3.40,
                estimated_cold_start_sec=15.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="modal",
                instance_type="modal.h100",
                gpu_spec=gpu_specs["H100"],
                price_per_hour=4.76,
                estimated_cold_start_sec=18.0,
                billing_granularity_sec=1,
            ),

            # --- RunPod (Dedicated IaaS, cheap spot instances) ---
            InstanceOffer(
                provider="runpod",
                instance_type="runpod.rtx3090",
                gpu_spec=gpu_specs["RTX3090"],
                price_per_hour=0.44,
                spot_price_per_hour=0.29,
                estimated_cold_start_sec=65.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="runpod",
                instance_type="runpod.rtx4090",
                gpu_spec=gpu_specs["RTX4090"],
                price_per_hour=0.69,
                spot_price_per_hour=0.44,
                estimated_cold_start_sec=65.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="runpod",
                instance_type="runpod.a10g",
                gpu_spec=gpu_specs["A10G"],
                price_per_hour=0.70,
                spot_price_per_hour=0.40,
                estimated_cold_start_sec=75.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="runpod",
                instance_type="runpod.l4",
                gpu_spec=gpu_specs["L4"],
                price_per_hour=0.70,
                spot_price_per_hour=0.40,
                estimated_cold_start_sec=75.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="runpod",
                instance_type="runpod.a100-80gb",
                gpu_spec=gpu_specs["A100-80GB"],
                price_per_hour=2.49,
                spot_price_per_hour=1.89,
                estimated_cold_start_sec=90.0,
                billing_granularity_sec=1,
            ),
            InstanceOffer(
                provider="runpod",
                instance_type="runpod.h100",
                gpu_spec=gpu_specs["H100"],
                price_per_hour=3.89,
                spot_price_per_hour=3.29,
                estimated_cold_start_sec=110.0,
                billing_granularity_sec=1,
            ),

            # --- Lambda Labs ---
            InstanceOffer(
                provider="lambda",
                instance_type="lambda.1x-a10",
                gpu_spec=gpu_specs["A10"],
                price_per_hour=0.60,
                spot_price_per_hour=None,
                estimated_cold_start_sec=60.0,
            ),
            InstanceOffer(
                provider="lambda",
                instance_type="lambda.1x-a100-40gb",
                gpu_spec=gpu_specs["A100-40GB"],
                price_per_hour=1.10,
                spot_price_per_hour=None,
                estimated_cold_start_sec=75.0,
            ),
            InstanceOffer(
                provider="lambda",
                instance_type="lambda.1x-a100-80gb",
                gpu_spec=gpu_specs["A100-80GB"],
                price_per_hour=1.49,
                spot_price_per_hour=None,
                estimated_cold_start_sec=80.0,
            ),
            InstanceOffer(
                provider="lambda",
                instance_type="lambda.1x-h100",
                gpu_spec=gpu_specs["H100"],
                price_per_hour=2.49,
                spot_price_per_hour=None,
                estimated_cold_start_sec=90.0,
            ),

            # --- AWS ---
            InstanceOffer(
                provider="aws",
                instance_type="g4dn.xlarge",
                gpu_spec=gpu_specs["T4"],
                price_per_hour=0.526,
                spot_price_per_hour=0.158,
                estimated_cold_start_sec=120.0,
            ),
            InstanceOffer(
                provider="aws",
                instance_type="g5.xlarge",
                gpu_spec=gpu_specs["A10G"],
                price_per_hour=1.006,
                spot_price_per_hour=0.402,
                estimated_cold_start_sec=120.0,
            ),
            InstanceOffer(
                provider="aws",
                instance_type="g6.xlarge",
                gpu_spec=gpu_specs["L4"],
                price_per_hour=0.805,
                spot_price_per_hour=0.322,
                estimated_cold_start_sec=120.0,
            ),

            # --- Nebius ---
            InstanceOffer(
                provider="nebius",
                instance_type="nebius.l4",
                gpu_spec=gpu_specs["L4"],
                price_per_hour=0.65,
                spot_price_per_hour=0.35,
                estimated_cold_start_sec=80.0,
            ),
            InstanceOffer(
                provider="nebius",
                instance_type="nebius.a100-80gb",
                gpu_spec=gpu_specs["A100-80GB"],
                price_per_hour=2.10,
                spot_price_per_hour=1.40,
                estimated_cold_start_sec=90.0,
            ),

            # --- Vast.ai ---
            InstanceOffer(
                provider="vast",
                instance_type="vast.rtx4090",
                gpu_spec=gpu_specs["RTX4090"],
                price_per_hour=0.38,
                spot_price_per_hour=0.25,
                estimated_cold_start_sec=90.0,
            ),
            InstanceOffer(
                provider="vast",
                instance_type="vast.a100-80gb",
                gpu_spec=gpu_specs["A100-80GB"],
                price_per_hour=1.40,
                spot_price_per_hour=0.99,
                estimated_cold_start_sec=110.0,
            ),
        ]

        return offers
