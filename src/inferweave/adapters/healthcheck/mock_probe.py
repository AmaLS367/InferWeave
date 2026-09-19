"""Mock health probe adapter for tests and dry-run simulations."""

from collections.abc import Sequence

from inferweave.domain.healthcheck import ProbeOutcome, ProbeResult
from inferweave.ports.healthcheck import HealthcheckProbePort


class MockHealthcheckProbeAdapter(HealthcheckProbePort):
    """In-memory mock probe adapter with configurable sequential outcomes."""

    def __init__(
        self,
        canned_results: Sequence[ProbeResult] | None = None,
        default_healthy: bool = True,
    ) -> None:
        self._canned_results = list(canned_results) if canned_results else []
        self._default_healthy = default_healthy
        self.probed_urls: list[str] = []
        self.probe_call_count = 0
        self.is_closed = False

    def enqueue_result(self, result: ProbeResult) -> None:
        """Adds a result to the queue of canned probe outcomes."""
        self._canned_results.append(result)

    async def probe(
        self,
        url: str,
        method: str = "GET",
        timeout_seconds: float = 5.0,
        headers: dict[str, str] | None = None,
        expected_status_codes: list[int] | None = None,
    ) -> ProbeResult:
        """Returns the next queued result or a default healthy/unhealthy result."""
        self.probe_call_count += 1
        self.probed_urls.append(url)

        if self._canned_results:
            return self._canned_results.pop(0)

        if self._default_healthy:
            return ProbeResult(
                is_healthy=True,
                outcome=ProbeOutcome.SUCCESS,
                status_code=200,
                latency_ms=10.0,
            )
        return ProbeResult(
            is_healthy=False,
            outcome=ProbeOutcome.CONNECTION_REFUSED,
            status_code=None,
            latency_ms=5.0,
            error_message="Mock connection refused",
        )

    async def close(self) -> None:
        self.is_closed = True
