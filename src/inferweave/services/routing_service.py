"""Application use case service orchestrating intelligent provider resolution."""

import asyncio
import logging
import sys

from inferweave.adapters.catalog.static_catalog import StaticCatalogAdapter
from inferweave.core.exceptions import NoFeasibleProviderError
from inferweave.models.deployment import DeploymentRequest
from inferweave.models.profile import ModelProfile
from inferweave.models.routing import (
    InstanceOffer,
    RoutingConstraints,
    RoutingDecision,
)
from inferweave.ports.catalog import AvailabilityProbePort, ProviderCatalogPort
from inferweave.routing.strategies import get_strategy

logger = logging.getLogger(__name__)


class SmartRoutingService:
    """Orchestrates candidate filtering, availability verification, and strategy ranking."""

    def __init__(
        self,
        catalog: ProviderCatalogPort | None = None,
        availability_probe: AvailabilityProbePort | None = None,
    ) -> None:
        self._catalog = catalog or StaticCatalogAdapter()
        self._probe = availability_probe

    async def aresolve(
        self,
        profile: ModelProfile,
        request: DeploymentRequest,
    ) -> RoutingDecision:
        """Asynchronously selects the best provider and instance offer for the deployment request."""
        constraints = self._build_constraints(profile, request)
        all_offers = await self._catalog.get_offers()

        # Step 1: Filter candidates based on hard requirements
        feasible_offers, rejection_reasons = self._filter_offers(all_offers, constraints, request)

        if not feasible_offers:
            reasons_summary = "\n  - " + "\n  - ".join(rejection_reasons)
            raise NoFeasibleProviderError(
                f"No compute provider satisfies requirements for model '{profile.id}' "
                f"(min VRAM: {constraints.min_vram_gb}GB, requested GPU: {constraints.gpu_type or 'any'})."
                f"{reasons_summary}",
                reasons=rejection_reasons,
            )

        # Step 2: Rank according to the requested strategy
        strategy = get_strategy(request.strategy)
        ranked = strategy.rank(feasible_offers, constraints)

        if not ranked:
            raise NoFeasibleProviderError(
                f"Strategy '{strategy.name}' produced no valid candidate rankings for model '{profile.id}'."
            )

        best = ranked[0]
        logger.info(
            "Auto-routing selected provider '%s' (%s, %s) using '%s' strategy for '%s'",
            best.offer.provider,
            best.offer.instance_type,
            best.reasoning,
            strategy.name,
            profile.id,
        )

        return RoutingDecision(
            chosen_provider=best.offer.provider,
            chosen_offer=best.offer,
            rankings=ranked,
            strategy_used=strategy.name,
        )

    def resolve(
        self,
        profile: ModelProfile,
        request: DeploymentRequest,
    ) -> RoutingDecision:
        """Synchronous wrapper for aresolve."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In an active event loop, run directly via future or new task in worker thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(lambda: asyncio.run(self.aresolve(profile, request))).result()

        return asyncio.run(self.aresolve(profile, request))

    def _build_constraints(
        self,
        profile: ModelProfile,
        request: DeploymentRequest,
    ) -> RoutingConstraints:
        allow_spot = request.custom_args.get("allow_spot", True)
        max_price = request.custom_args.get("max_price_per_hour", None)
        regions = request.custom_args.get("preferred_regions", [])

        return RoutingConstraints(
            min_vram_gb=profile.hardware.min_vram_gb,
            gpu_count=request.num_gpus or profile.hardware.gpu_count,
            gpu_type=request.gpu_type,
            recommended_gpus=profile.hardware.recommended_gpus,
            allow_spot=allow_spot,
            max_price_per_hour=max_price,
            preferred_regions=regions,
        )

    def _filter_offers(
        self,
        offers: list[InstanceOffer],
        constraints: RoutingConstraints,
        request: DeploymentRequest,
    ) -> tuple[list[InstanceOffer], list[str]]:
        feasible: list[InstanceOffer] = []
        reasons: list[str] = []

        # Check native Windows platform limitation for SkyPilot (when not dry_run)
        is_windows = sys.platform == "win32" and not request.dry_run

        for offer in offers:
            # 1. Platform compatibility: On native Windows, non-Modal providers (SkyPilot) cannot launch live clusters
            if is_windows and offer.provider != "modal":
                reasons.append(f"Provider '{offer.provider}' requires POSIX system calls (SkyPilot) on native Windows")
                continue

            # 2. Strict GPU model filter if specified
            if constraints.gpu_type:
                target_gpu = constraints.gpu_type.lower()
                offer_gpu = offer.gpu_spec.name.lower()
                if target_gpu not in offer_gpu and offer_gpu not in target_gpu:
                    reasons.append(f"Instance '{offer.instance_type}' GPU '{offer.gpu_spec.name}' does not match requested '{constraints.gpu_type}'")
                    continue

            # 3. Total VRAM capacity check
            if offer.total_vram_gb < constraints.min_vram_gb:
                reasons.append(f"Instance '{offer.instance_type}' VRAM {offer.total_vram_gb:.0f}GB is below required {constraints.min_vram_gb:.0f}GB")
                continue

            # 4. GPU count check
            if offer.gpu_count < constraints.gpu_count:
                reasons.append(f"Instance '{offer.instance_type}' has {offer.gpu_count} GPU(s), needed {constraints.gpu_count}")
                continue

            # 5. Availability check
            if not offer.is_available:
                reasons.append(f"Instance '{offer.instance_type}' on '{offer.provider}' is currently out of stock")
                continue

            # 6. Max price check if configured
            if constraints.max_price_per_hour is not None:
                eff_price = offer.effective_price(allow_spot=constraints.allow_spot)
                if eff_price > constraints.max_price_per_hour:
                    reasons.append(f"Instance '{offer.instance_type}' price ${eff_price:.2f}/hr exceeds limit ${constraints.max_price_per_hour:.2f}/hr")
                    continue

            feasible.append(offer)

        # Deduplicate reasons for clean reporting
        unique_reasons = list(dict.fromkeys(reasons))
        return feasible, unique_reasons
