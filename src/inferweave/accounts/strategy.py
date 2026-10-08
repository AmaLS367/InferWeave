"""Selection strategy plumbing: named strategies plus per-call exclusion and pinning."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from credweave import (
    CredentialCandidate,
    FailoverStrategy,
    LeastRecentlyUsedStrategy,
    LeastUsedStrategy,
    RandomStrategy,
    RoundRobinStrategy,
    SelectionContext,
    SelectionStrategy,
    WeightedStrategy,
)

from inferweave.core.exceptions import AccountConfigurationError

STRATEGIES: dict[str, Callable[[], SelectionStrategy]] = {
    "round_robin": RoundRobinStrategy,
    "weighted": WeightedStrategy,
    "failover": FailoverStrategy,
    "least_used": LeastUsedStrategy,
    "least_recently_used": LeastRecentlyUsedStrategy,
    "random": RandomStrategy,
}
_ALIASES = {"lru": "least_recently_used", "priority": "failover", "round-robin": "round_robin"}


def make_strategy(strategy: str | SelectionStrategy) -> SelectionStrategy:
    """Builds a fresh strategy from its name, or returns a caller-supplied instance."""
    if not isinstance(strategy, str):
        return strategy
    name = _ALIASES.get(strategy.strip().lower(), strategy.strip().lower())
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise AccountConfigurationError(
            f"Unknown account selection strategy '{strategy}'. Choose one of: "
            f"{', '.join(STRATEGIES)}."
        ) from None


@dataclass(frozen=True, repr=False)
class ScopedSelectionContext(SelectionContext):
    """A CredWeave selection context that can exclude or pin account ids for one acquire."""

    exclude: frozenset[str] = field(default_factory=frozenset)
    pin: str | None = None


class ScopedStrategy:
    """Wraps a CredWeave strategy so a single acquire can skip or require specific accounts.

    Exclusion keeps a failover loop from re-selecting an account that just failed for a reason
    that is not a credential fault (e.g. GPU stock-out). Pinning lets a caller request one
    account explicitly. Health, cooldown and concurrency rules still apply: CredWeave reserves
    the lease atomically, so a pinned account that is cooling down is simply unavailable.
    """

    def __init__(self, inner: SelectionStrategy) -> None:
        self._inner = inner

    @property
    def inner(self) -> SelectionStrategy:
        return self._inner

    @property
    def name(self) -> str:
        return self._inner.name

    def select(
        self,
        candidates: Sequence[CredentialCandidate],
        context: SelectionContext | None = None,
    ) -> CredentialCandidate | None:
        if isinstance(context, ScopedSelectionContext):
            if context.pin is not None:
                candidates = [c for c in candidates if c.credential_id == context.pin]
            if context.exclude:
                candidates = [c for c in candidates if c.credential_id not in context.exclude]
        return self._inner.select(candidates, context)
