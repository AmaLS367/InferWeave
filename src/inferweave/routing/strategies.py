"""Domain strategies for ranking and scoring candidate compute provider instance offers."""

from abc import ABC, abstractmethod

from inferweave.models.routing import InstanceOffer, RankedOffer, RoutingConstraints


class RoutingStrategy(ABC):
    """Abstract routing strategy defining how candidate instance offers are ranked."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy, e.g. 'cheapest', 'free_first', 'lowest_latency', 'balanced'."""

    @abstractmethod
    def rank(
        self,
        offers: list[InstanceOffer],
        constraints: RoutingConstraints,
    ) -> list[RankedOffer]:
        """Evaluates and orders candidate offers from best to worst."""


class CheapestStrategy(RoutingStrategy):
    """Ranks offers strictly by lowest hourly cost ($/hr)."""

    @property
    def name(self) -> str:
        return "cheapest"

    def rank(
        self,
        offers: list[InstanceOffer],
        constraints: RoutingConstraints,
    ) -> list[RankedOffer]:
        if not offers:
            return []

        # Sort primarily by effective price, then by lower cold start, then higher VRAM
        def sort_key(o: InstanceOffer) -> tuple[float, float, float]:
            price = o.effective_price(allow_spot=constraints.allow_spot)
            return (price, o.estimated_cold_start_sec, -o.total_vram_gb)

        sorted_offers = sorted(offers, key=sort_key)
        ranked: list[RankedOffer] = []

        for rank_idx, offer in enumerate(sorted_offers):
            price = offer.effective_price(allow_spot=constraints.allow_spot)
            cost_score = 1.0 / (1.0 + price)
            latency_score = 1.0 / (1.0 + offer.estimated_cold_start_sec / 10.0)
            avail_score = 1.0 if offer.is_available else 0.0

            # Rank-decay total score to preserve strict sort order in results
            total_score = max(0.0, 1.0 - (rank_idx * 0.05))

            spot_note = (
                " (Spot)"
                if constraints.allow_spot and offer.spot_price_per_hour is not None
                else " (On-Demand)"
            )
            reasoning = f"${price:.2f}/hr{spot_note}, {offer.total_vram_gb:.0f}GB VRAM, cold start ~{offer.estimated_cold_start_sec:.0f}s"

            ranked.append(
                RankedOffer(
                    offer=offer,
                    total_score=total_score,
                    cost_score=cost_score,
                    latency_score=latency_score,
                    availability_score=avail_score,
                    reasoning=reasoning,
                )
            )

        return ranked


class FreeFirstStrategy(RoutingStrategy):
    """Prioritizes free tiers and zero-cost options, falling back to cheapest."""

    @property
    def name(self) -> str:
        return "free_first"

    def rank(
        self,
        offers: list[InstanceOffer],
        constraints: RoutingConstraints,
    ) -> list[RankedOffer]:
        if not offers:
            return []

        def sort_key(o: InstanceOffer) -> tuple[int, float, float]:
            price = o.effective_price(allow_spot=constraints.allow_spot)
            is_free = o.is_free_tier or price == 0.0
            # 0 for free first, 1 for paid
            tier_flag = 0 if is_free else 1
            return (tier_flag, price, o.estimated_cold_start_sec)

        sorted_offers = sorted(offers, key=sort_key)
        ranked: list[RankedOffer] = []

        for rank_idx, offer in enumerate(sorted_offers):
            price = offer.effective_price(allow_spot=constraints.allow_spot)
            is_free = offer.is_free_tier or price == 0.0
            cost_score = 1.0 if is_free else (1.0 / (1.0 + price))
            latency_score = 1.0 / (1.0 + offer.estimated_cold_start_sec / 10.0)
            avail_score = 1.0 if offer.is_available else 0.0

            tier_boost = 1.0 if is_free else 0.0
            total_score = tier_boost + max(0.0, 0.5 - (rank_idx * 0.02))

            tag = "Free Tier" if is_free else f"${price:.2f}/hr"
            reasoning = f"{tag}, {offer.total_vram_gb:.0f}GB VRAM on {offer.provider}"

            ranked.append(
                RankedOffer(
                    offer=offer,
                    total_score=total_score,
                    cost_score=cost_score,
                    latency_score=latency_score,
                    availability_score=avail_score,
                    reasoning=reasoning,
                )
            )

        return ranked


class LowestLatencyStrategy(RoutingStrategy):
    """Prioritizes fastest cold start and high-performance accelerator architectures."""

    @property
    def name(self) -> str:
        return "lowest_latency"

    def rank(
        self,
        offers: list[InstanceOffer],
        constraints: RoutingConstraints,
    ) -> list[RankedOffer]:
        if not offers:
            return []

        def sort_key(o: InstanceOffer) -> tuple[float, float]:
            # Lowest cold start first, then lowest price as tie-breaker
            price = o.effective_price(allow_spot=constraints.allow_spot)
            return (o.estimated_cold_start_sec, price)

        sorted_offers = sorted(offers, key=sort_key)
        ranked: list[RankedOffer] = []

        for rank_idx, offer in enumerate(sorted_offers):
            price = offer.effective_price(allow_spot=constraints.allow_spot)
            latency_score = 1.0 / (1.0 + offer.estimated_cold_start_sec / 5.0)
            cost_score = 1.0 / (1.0 + price)
            avail_score = 1.0 if offer.is_available else 0.0

            total_score = max(0.0, 1.0 - (rank_idx * 0.05))
            reasoning = f"Cold start ~{offer.estimated_cold_start_sec:.0f}s, ${price:.2f}/hr on {offer.provider}"

            ranked.append(
                RankedOffer(
                    offer=offer,
                    total_score=total_score,
                    cost_score=cost_score,
                    latency_score=latency_score,
                    availability_score=avail_score,
                    reasoning=reasoning,
                )
            )

        return ranked


class BalancedStrategy(RoutingStrategy):
    """Multi-objective Pareto-like ranking combining cost (40%), latency (35%), and availability (25%)."""

    @property
    def name(self) -> str:
        return "balanced"

    def rank(
        self,
        offers: list[InstanceOffer],
        constraints: RoutingConstraints,
    ) -> list[RankedOffer]:
        if not offers:
            return []

        prices = [o.effective_price(allow_spot=constraints.allow_spot) for o in offers]
        latencies = [o.estimated_cold_start_sec for o in offers]

        min_p, max_p = min(prices), max(prices)
        min_l, max_l = min(latencies), max(latencies)

        scored: list[tuple[float, float, float, float, InstanceOffer]] = []

        for o in offers:
            p = o.effective_price(allow_spot=constraints.allow_spot)
            l = o.estimated_cold_start_sec

            # Normalize [0.0 - 1.0] where 1.0 is best
            c_score = 1.0 if max_p == min_p else (max_p - p) / (max_p - min_p)
            l_score = 1.0 if max_l == min_l else (max_l - l) / (max_l - min_l)
            a_score = 1.0 if o.is_available else 0.0

            tot_score = (0.40 * c_score) + (0.35 * l_score) + (0.25 * a_score)
            scored.append((tot_score, c_score, l_score, a_score, o))

        # Sort descending by total score
        scored.sort(key=lambda item: item[0], reverse=True)

        ranked: list[RankedOffer] = []
        for tot, c, l, a, offer in scored:
            p = offer.effective_price(allow_spot=constraints.allow_spot)
            reasoning = f"Balanced score {tot:.2f} (cost: {c:.2f}, lat: {l:.2f}, avail: {a:.2f}, ${p:.2f}/hr)"
            ranked.append(
                RankedOffer(
                    offer=offer,
                    total_score=tot,
                    cost_score=c,
                    latency_score=l,
                    availability_score=a,
                    reasoning=reasoning,
                )
            )

        return ranked


STRATEGY_REGISTRY: dict[str, RoutingStrategy] = {
    "cheapest": CheapestStrategy(),
    "free_first": FreeFirstStrategy(),
    "lowest_latency": LowestLatencyStrategy(),
    "balanced": BalancedStrategy(),
}


def get_strategy(strategy_name: str | None) -> RoutingStrategy:
    """Retrieves a routing strategy by name, defaulting to 'cheapest'."""
    key = (strategy_name or "cheapest").lower()
    if key not in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY["cheapest"]
    return STRATEGY_REGISTRY[key]
