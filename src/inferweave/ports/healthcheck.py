"""Port definition for endpoint health probes."""

from abc import ABC, abstractmethod

from inferweave.domain.healthcheck import ProbeResult


class HealthcheckProbePort(ABC):
    """Abstract port for executing HTTP/network probes against remote deployment endpoints."""

    @abstractmethod
    async def probe(
        self,
        url: str,
        method: str = "GET",
        timeout_seconds: float = 5.0,
        headers: dict[str, str] | None = None,
        expected_status_codes: list[int] | None = None,
    ) -> ProbeResult:
        """Executes a single probe request against target endpoint URL.

        Args:
            url: Full HTTP/HTTPS probe URL (e.g. 'http://1.2.3.4:8000/health').
            method: HTTP method, usually 'GET' or 'HEAD'.
            timeout_seconds: Timeout ceiling for this individual probe attempt.
            headers: Optional dictionary of HTTP headers.
            expected_status_codes: Optional list of acceptable HTTP status codes.

        Returns:
            ProbeResult: Standardized result containing health outcome, latency, and status code.
        """

    @abstractmethod
    async def close(self) -> None:
        """Cleans up active HTTP sessions or connection pools."""
