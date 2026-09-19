"""HTTPX-backed health probe adapter."""

import logging
import time
from types import TracebackType
from typing import Self

import httpx

from inferweave.domain.healthcheck import HealthEvaluator, ProbeResult
from inferweave.ports.healthcheck import HealthcheckProbePort

logger = logging.getLogger(__name__)


class HttpxHealthcheckProbeAdapter(HealthcheckProbePort):
    """Executes non-blocking HTTP probes using HTTPX with persistent connection reuse."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        evaluator: HealthEvaluator | None = None,
        verify_ssl: bool = True,
    ) -> None:
        self._custom_client = client is not None
        self._client = client
        self._evaluator = evaluator or HealthEvaluator()
        self._verify_ssl = verify_ssl

    def _get_client(self) -> httpx.AsyncClient:
        if self._custom_client and self._client is not None:
            return self._client
        is_closed = getattr(self._client, "is_closed", False)
        if isinstance(is_closed, bool) and is_closed:
            self._client = None
        if self._client is None:
            self._client = httpx.AsyncClient(
                verify=self._verify_ssl,
                follow_redirects=True,
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        return self._client

    async def probe(
        self,
        url: str,
        method: str = "GET",
        timeout_seconds: float = 5.0,
        headers: dict[str, str] | None = None,
        expected_status_codes: list[int] | None = None,
    ) -> ProbeResult:
        """Executes a single HTTP probe against the specified URL."""
        client = self._get_client()
        start = time.perf_counter()

        timeout = httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 3.0))

        try:
            response = await client.request(
                method=method.upper(),
                url=url,
                timeout=timeout,
                headers=headers or {},
            )
            latency = (time.perf_counter() - start) * 1000.0
            return self._evaluator.evaluate_probe(
                status_code=response.status_code,
                latency_ms=latency,
                expected_status_codes=expected_status_codes,
            )
        except httpx.TimeoutException as err:
            latency = (time.perf_counter() - start) * 1000.0
            logger.debug("Probe to '%s' timed out after %.2fms: %s", url, latency, err)
            return self._evaluator.evaluate_probe(
                latency_ms=latency,
                error=err,
                expected_status_codes=expected_status_codes,
            )
        except httpx.ConnectError as err:
            latency = (time.perf_counter() - start) * 1000.0
            logger.debug(
                "Probe to '%s' failed to connect (%.2fms): %s", url, latency, err
            )
            return self._evaluator.evaluate_probe(
                latency_ms=latency,
                error=err,
                expected_status_codes=expected_status_codes,
            )
        except httpx.HTTPError as err:
            latency = (time.perf_counter() - start) * 1000.0
            logger.debug("Probe to '%s' HTTP error (%.2fms): %s", url, latency, err)
            return self._evaluator.evaluate_probe(
                latency_ms=latency,
                error=err,
                expected_status_codes=expected_status_codes,
            )
        except Exception as err:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000.0
            logger.debug(
                "Probe to '%s' unexpected error (%.2fms): %s", url, latency, err
            )
            return self._evaluator.evaluate_probe(
                latency_ms=latency,
                error=err,
                expected_status_codes=expected_status_codes,
            )

    async def close(self) -> None:
        """Closes the underlying HTTP client if managed internally."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
